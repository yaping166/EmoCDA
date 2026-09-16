from __future__ import annotations

import json
from typing import Callable, Dict, List

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from utils.metrics import Pair, fold_category

INPUT_PROMPT = (
    "Aspect Category Sentiment Analysis.\n"
    "Given a sentence, identify all mentioned aspect categories and determine "
    "their sentiment.\n"
    "Sentence: "
)

_VAL_RATIO = 0.1


def _dedup_pairs(
    labels: List[Dict[str, str]],
    *,
    value_key: str,
    granularity: str,
) -> List[Pair]:
    seen: set = set()
    pairs: List[Pair] = []
    for lab in labels:
        cat = (lab.get("category") or "").strip()
        val = (lab.get(value_key) or "").strip()
        if not cat or not val:
            continue
        key = (fold_category(cat, granularity), val)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def _pairs_to_text(pairs: List[Pair]) -> str:
    return ", ".join(f"{c}: {v}" for c, v in pairs)


def prepare_acsa_emotion_dataset(
    train_path: str,
    test_path: str,
    *,
    granularity: str = "EA",
    seed: int = 42,
) -> DatasetDict:
    def convert(example: Dict) -> Dict[str, str]:
        text = example.get("text", "")
        labels = example.get("labels") or []
        return {
            "input": text,
            "main_output": _pairs_to_text(
                _dedup_pairs(labels, value_key="sentiment", granularity=granularity)
            ),
            "aux_output": _pairs_to_text(
                _dedup_pairs(labels, value_key="emotion", granularity=granularity)
            ),
        }

    with open(train_path, encoding="utf-8") as f:
        train_ds = Dataset.from_list([convert(x) for x in json.load(f)])
    with open(test_path, encoding="utf-8") as f:
        test_ds = Dataset.from_list([convert(x) for x in json.load(f)])

    split = train_ds.train_test_split(test_size=_VAL_RATIO, seed=seed)
    train_ds, val_ds = split["train"], split["test"]

    return DatasetDict({"train": train_ds, "valid": val_ds, "test": test_ds})


def build_tokenize_function(
    tokenizer: PreTrainedTokenizerBase,
    max_input_length: int,
    max_output_length: int,
) -> Callable[[Dict[str, List[str]]], Dict]:
    def tokenize_batch(examples):
        model_inputs = tokenizer(
            [INPUT_PROMPT + t for t in examples["input"]],
            max_length=max_input_length,
            truncation=True,
        )
        main_labels = tokenizer(
            text_target=examples["main_output"],
            max_length=max_output_length,
            truncation=True,
        )
        aux_labels = tokenizer(
            text_target=examples["aux_output"],
            max_length=max_output_length,
            truncation=True,
        )
        model_inputs["labels"] = main_labels["input_ids"]
        model_inputs["aux_labels"] = aux_labels["input_ids"]
        return model_inputs

    return tokenize_batch
