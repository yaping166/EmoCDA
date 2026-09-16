#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from transformers import AutoTokenizer

from utils.data import build_tokenize_function, prepare_acsa_emotion_dataset
from utils.metrics import GRANULARITIES, compute_metrics_factory
from training.pipeline import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ACSA–ACEA dual-decoder MTL with bidirectional cross-attention.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = parser.add_argument_group("I/O")
    io.add_argument("--train_path", required=True)
    io.add_argument("--test_path", required=True)
    io.add_argument("--dataset_name", default="rest15")
    io.add_argument("--output_base", default="run")

    task = parser.add_argument_group("Task setup")
    task.add_argument(
        "--granularity",
        choices=list(GRANULARITIES),
        default="EA",
        help="Category granularity.",
    )
    task.add_argument("--model_name", default="google/flan-t5-large")
    task.add_argument(
        "--gamma",
        type=float,
        default=0.6,
        help="output = main_hidden + gamma * C.",
    )
    task.add_argument(
        "--warmup_epochs",
        type=float,
        default=5.0,
        help="Epochs with cross-attention fully disabled.",
    )

    train = parser.add_argument_group("Training")
    train.add_argument("--batch_size", type=int, default=4)
    train.add_argument("--learning_rate", type=float, default=3e-5)
    train.add_argument("--num_train_epochs", type=int, default=10)
    train.add_argument("--gradient_accumulation_steps", type=int, default=1)
    train.add_argument("--max_input_length", type=int, default=256)
    train.add_argument("--max_output_length", type=int, default=128)
    train.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument("--run", type=int, default=42, help="Random seed.")

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    datasets = prepare_acsa_emotion_dataset(
        args.train_path,
        args.test_path,
        granularity=args.granularity,
        seed=args.run,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenize_fn = build_tokenize_function(
        tokenizer, args.max_input_length, args.max_output_length
    )
    tokenized = datasets.map(
        tokenize_fn,
        remove_columns=["input", "main_output", "aux_output"],
        batched=True,
    )

    compute_metrics = compute_metrics_factory(tokenizer, args.granularity)
    train_and_evaluate(args, args.run, tokenizer, tokenized, compute_metrics)


if __name__ == "__main__":
    main()
