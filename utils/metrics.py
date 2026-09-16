from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from transformers import PreTrainedTokenizerBase

from utils.decoding import decode_token_ids, split_main_aux

Pair = Tuple[str, str]

GRANULARITIES = ("EA", "E")


def fold_category(category: str, granularity: str) -> str:
    if granularity == "E" and "#" in category:
        return category.split("#", 1)[0]
    return category


def fold_pairs(pairs: Iterable[Pair], granularity: str) -> List[Pair]:
    seen: set = set()
    out: List[Pair] = []
    for cat, lbl in pairs:
        key = (fold_category(cat, granularity), lbl)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def parse_pairs_text(text: str) -> List[Pair]:
    text = text.strip()
    if not text:
        return []
    pairs: List[Pair] = []
    for segment in text.split(","):
        segment = segment.strip()
        if ":" not in segment:
            continue
        cat, lbl = segment.split(":", 1)
        cat, lbl = cat.strip(), lbl.strip()
        if cat and lbl:
            pairs.append((cat, lbl))
    return pairs


@dataclass(frozen=True)
class PRF1:
    precision: float
    recall: float
    f1: float
    tp: int
    n_pred: int
    n_gold: int

    def as_dict(self, ndigits: int = 4) -> dict:
        return {
            "precision": round(self.precision, ndigits),
            "recall": round(self.recall, ndigits),
            "f1": round(self.f1, ndigits),
        }


def micro_prf(
    predicted_pairs: Sequence[Iterable[Pair]],
    true_pairs: Sequence[Iterable[Pair]],
) -> PRF1:
    if len(predicted_pairs) != len(true_pairs):
        raise ValueError(
            f"predicted and true pair lists must have the same length "
            f"({len(predicted_pairs)} vs {len(true_pairs)})"
        )
    tp = n_pred = n_gold = 0
    for preds, golds in zip(predicted_pairs, true_pairs):
        pred_set, gold_set = set(preds), set(golds)
        tp += len(pred_set & gold_set)
        n_pred += len(pred_set)
        n_gold += len(gold_set)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return PRF1(precision, recall, f1, tp, n_pred, n_gold)


def decode_to_folded_pairs(
    tokenizer: PreTrainedTokenizerBase,
    raw_predictions,
    granularity: str,
) -> List[List[Pair]]:
    return [
        fold_pairs(parse_pairs_text(text), granularity)
        for text in decode_token_ids(tokenizer, raw_predictions)
    ]


def compute_metrics_factory(tokenizer: PreTrainedTokenizerBase, granularity: str):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        main_pred, aux_pred = split_main_aux(predictions)
        main_lbl, aux_lbl = split_main_aux(labels)

        main = micro_prf(
            decode_to_folded_pairs(tokenizer, main_pred, granularity),
            decode_to_folded_pairs(tokenizer, main_lbl, granularity),
        )
        results = {"precision": main.precision, "recall": main.recall, "f1": main.f1}

        if aux_pred is not None and aux_lbl is not None:
            aux = micro_prf(
                decode_to_folded_pairs(tokenizer, aux_pred, granularity),
                decode_to_folded_pairs(tokenizer, aux_lbl, granularity),
            )
            results.update(
                {
                    "aux_precision": aux.precision,
                    "aux_recall": aux.recall,
                    "aux_f1": aux.f1,
                }
            )
        return results

    return compute_metrics
