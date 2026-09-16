from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import Seq2SeqTrainingArguments
from transformers.trainer_callback import TrainerCallback

from utils.decoding import split_main_aux
from utils.metrics import PRF1, Pair, decode_to_folded_pairs, fold_pairs, micro_prf
from model.cross_decoder_attention import comma_token_ids
from model.dual_decoder import DualDecoderMTLModel
from training.collator import DualDecoderDataCollator
from training.trainer import DualDecoderTrainer

logger = logging.getLogger(__name__)

BEST_METRIC_NAME = "eval_f1"


def _init_seed(seed: int) -> None:
    from transformers import set_seed

    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class BestModelSaver(TrainerCallback):
    def __init__(self, best_dir: str, metric_name: str, tokenizer, min_epoch: float = 0.0):
        self.best_dir = best_dir
        self.metric_name = metric_name
        self.tokenizer = tokenizer
        self.min_epoch = min_epoch
        self.best_metric = float("-inf")
        self._trainer = None

    def bind_trainer(self, trainer) -> None:
        self._trainer = trainer

    def on_evaluate(self, args, state, control, **kwargs) -> None:
        metrics = kwargs.get("metrics") or {}
        if self.metric_name not in metrics:
            return
        value = float(metrics[self.metric_name])
        if value <= self.best_metric:
            return
        if (state.epoch or 0.0) < self.min_epoch:
            return
        self.best_metric = value
        if os.path.exists(self.best_dir):
            shutil.rmtree(self.best_dir)
        os.makedirs(self.best_dir, exist_ok=True)
        self._trainer.save_model(self.best_dir)
        self.tokenizer.save_pretrained(self.best_dir)


def _build_trainer(
    model,
    tokenizer,
    training_args: Seq2SeqTrainingArguments,
    *,
    train_dataset=None,
    eval_dataset=None,
    compute_metrics=None,
):
    return DualDecoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DualDecoderDataCollator(tokenizer=tokenizer, model=model),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )


@dataclass
class TrainResult:
    output_dir: str
    logging_dir: Optional[str]
    best_dir: str
    best_metric: float
    metric_name: str


def train_return_best_on_valid(
    args,
    run: int,
    tokenizer,
    tokenized_datasets,
    compute_metrics,
) -> TrainResult:
    _init_seed(run)
    model = DualDecoderMTLModel.create(
        args.model_name,
        gamma=args.gamma,
        warmup_epochs=args.warmup_epochs,
        separator_ids=comma_token_ids(tokenizer),
        tokenizer=tokenizer,
        align_granularity=args.granularity,
    )

    base_output = args.output_base
    output_dir = os.path.join(base_output, args.granularity, args.dataset_name)

    if os.path.exists(output_dir):
        logger.info("Removing stale checkpoint directory %s", output_dir)
        shutil.rmtree(output_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="no",
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        seed=run,
        data_seed=run,
        full_determinism=True,
        dataloader_num_workers=0,
        bf16=args.bf16,
        generation_max_length=args.max_output_length,
        generation_num_beams=4,
        load_best_model_at_end=False,
        prediction_loss_only=False,
        report_to="none",
    )

    trainer = _build_trainer(
        model,
        tokenizer,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        compute_metrics=compute_metrics,
    )

    best_dir = os.path.join(output_dir, "best_model")

    min_epoch = args.warmup_epochs + 2.0
    best_cb = BestModelSaver(best_dir, BEST_METRIC_NAME, tokenizer, min_epoch=min_epoch)
    best_cb.bind_trainer(trainer)
    trainer.add_callback(best_cb)
    trainer.train()

    if not os.path.exists(best_dir):
        trainer.save_model(best_dir)
        tokenizer.save_pretrained(best_dir)

    return TrainResult(
        output_dir=output_dir,
        logging_dir=None,
        best_dir=best_dir,
        best_metric=best_cb.best_metric,
        metric_name=BEST_METRIC_NAME,
    )


def _remove_trainer_scratch(trainer) -> None:
    output_dir = getattr(getattr(trainer, "args", None), "output_dir", None)
    if output_dir:
        shutil.rmtree(output_dir, ignore_errors=True)


def _make_test_eval_trainer(args, tokenizer, model_dir: str):
    model = DualDecoderMTLModel.from_pretrained_dual(
        model_dir, tokenizer=tokenizer
    )
    model.set_gamma_override(1.0)

    training_args = Seq2SeqTrainingArguments(
        tempfile.mkdtemp(prefix="acsa_predict_"),
        remove_unused_columns=False,
        eval_strategy="no",
        save_strategy="no",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=args.max_output_length,
        generation_num_beams=4,
        bf16=args.bf16,
        report_to="none",
    )
    trainer = _build_trainer(model, tokenizer, training_args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer.model.to(device)
    return trainer


def gold_pairs_from_raw(
    test_raw: List[dict],
    value_key: str,
    granularity: str,
) -> List[List[Pair]]:
    out: List[List[Pair]] = []
    for example in test_raw:
        labels = example.get("labels") or []
        pairs = []
        for label in labels:
            cat = (label.get("category") or "").strip()
            val = (label.get(value_key) or "").strip()
            if cat and val:
                pairs.append((cat, val))
        out.append(fold_pairs(pairs, granularity))
    return out


def _write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_detailed(
    path: str,
    raw_examples: List[dict],
    pred_pairs: List[List[Pair]],
    gold_pairs: List[List[Pair]],
) -> None:
    records = []
    for example, preds, golds in zip(raw_examples, pred_pairs, gold_pairs):
        records.append(
            {
                "text": example.get("text", ""),
                "predicted_aspects": ", ".join(f"{c}: {v}" for c, v in preds),
                "true_aspects": ", ".join(f"{c}: {v}" for c, v in golds),
                "is_correct": set(preds) == set(golds),
            }
        )
    _write_json(path, records)


def evaluate_prf1(
    trainer,
    tokenizer,
    tokenized_ds,
    raw_examples: List[dict],
    *,
    granularity: str,
) -> Tuple[PRF1, List[List[Pair]]]:
    predictions = trainer.predict(tokenized_ds).predictions
    main_raw, _aux_raw = split_main_aux(predictions)

    main_pred_pairs = decode_to_folded_pairs(tokenizer, main_raw, granularity)
    main_gold_pairs = gold_pairs_from_raw(raw_examples, "sentiment", granularity)
    main_metrics = micro_prf(main_pred_pairs, main_gold_pairs)

    return main_metrics, main_pred_pairs


def save_eval_artifacts(
    output_dir: str,
    raw_examples: List[dict],
    main_metrics: PRF1,
    main_pred_pairs: List[List[Pair]],
    *,
    granularity: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    _write_json(os.path.join(output_dir, "metrics.json"), main_metrics.as_dict())
    _write_detailed(
        os.path.join(output_dir, "detailed_results.json"),
        raw_examples,
        main_pred_pairs,
        gold_pairs_from_raw(raw_examples, "sentiment", granularity),
    )


def evaluate_and_save(
    trainer,
    tokenizer,
    tokenized_test_ds,
    test_raw_path: str,
    output_dir: str,
    *,
    granularity: str,
) -> PRF1:
    with open(test_raw_path, encoding="utf-8") as f:
        test_raw = json.load(f)
    main_metrics, main_pred_pairs = evaluate_prf1(
        trainer,
        tokenizer,
        tokenized_test_ds,
        test_raw,
        granularity=granularity,
    )
    save_eval_artifacts(
        output_dir,
        test_raw,
        main_metrics,
        main_pred_pairs,
        granularity=granularity,
    )
    return main_metrics


def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics) -> None:
    info = train_return_best_on_valid(
        args, run, tokenizer, tokenized_datasets, compute_metrics
    )
    if not os.path.exists(info.best_dir):
        raise RuntimeError(f"Training did not produce a best model at {info.best_dir}")

    trainer = _make_test_eval_trainer(args, tokenizer, info.best_dir)
    try:
        overall = evaluate_and_save(
            trainer,
            tokenizer,
            tokenized_datasets["test"],
            args.test_path,
            info.output_dir,
            granularity=args.granularity,
        )
    finally:
        _remove_trainer_scratch(trainer)

    logger.info("Final test metrics: %s", overall.as_dict())
    shutil.rmtree(info.best_dir, ignore_errors=True)
