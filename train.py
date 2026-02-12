#!/usr/bin/env python
import json
import numpy as np
import pandas as pd
from collections import Counter
import argparse
import logging
import os
import re
from pathlib import Path
from datetime import datetime, timezone

import boto3
from transformers import AutoTokenizer, TrainingArguments

from src.data_loader import prepare_data
from src.model import get_model
from src.trainer import WeightedCELossTrainer
from src.utils import compute_metrics

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

LOW_CONFIDENCE_THRESHOLD = 0.60
ERROR_TOP_K = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a financial NLP classifier with LoRA/PEFT.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ProsusAI/finbert",
        help="Base model name or path.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Per-device batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA/PEFT if set; otherwise standard fine-tuning.",
    )
    parser.add_argument(
        "--use_s3",
        action="store_true",
        help="If set, simulate loading/uploading artifacts to S3.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed precision training (FP16).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save model checkpoints and artifacts.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max sequence length for tokenization.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run identifier for exported evaluation logs.",
    )
    return parser.parse_args()


def _sanitize_model_name(model_name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "-", model_name).strip("-").lower()
    return sanitized or "model"


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _resolve_text_column(dataset_columns) -> str:
    for candidate in ("sentence", "text"):
        if candidate in dataset_columns:
            return candidate
    return ""


def upload_artifacts_to_s3(output_dir: str, bucket: str, prefix: str = "artifacts") -> None:
    """
    Upload all files in output_dir to the specified S3 bucket/prefix.
    """
    s3_client = boto3.client("s3")
    for root, _, files in os.walk(output_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, output_dir)
            s3_key = f"{prefix}/{rel_path}"
            logger.info("Uploading %s to s3://%s/%s", local_path, bucket, s3_key)
            s3_client.upload_file(local_path, bucket, s3_key)


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    auto_run_id = (
        f"{_sanitize_model_name(args.model_name)}-"
        f"lora{int(args.use_lora)}-fp16{int(args.fp16)}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )
    run_id = args.run_id if args.run_id else auto_run_id
    logger.info("Using run_id=%s", run_id)

    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    logger.info("Preparing data (financial_phrasebank)...")
    tokenized_ds, class_weights = prepare_data(
        tokenizer=tokenizer,
        subset="sentences_allagree",
        use_s3=args.use_s3,
        bucket=None,  # replace with your bucket if using S3 ingestion
        key=None,     # replace with your key if using S3 ingestion
        max_length=args.max_length,
    )

    num_labels = int(tokenized_ds["train"].to_pandas()["labels"].nunique())
    logger.info("Detected %d labels.", num_labels)

    logger.info("Initializing model (LoRA=%s)...", args.use_lora)
    model = get_model(
        model_name=args.model_name,
        num_labels=num_labels,
        use_lora=args.use_lora,
    )
    # ---- DEBUG: verify which params are trainable (important for LoRA) ----
    logger.info("=== Trainable parameters check ===")
    trainable = []
    head_trainable = []
    lora_trainable = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable.append(n)
            if "classifier" in n or "score" in n:
                head_trainable.append(n)
            if "lora" in n.lower():
                lora_trainable.append(n)

    logger.info("Trainable param count: %d", len(trainable))
    logger.info("Trainable head params (classifier/score): %s", head_trainable if head_trainable else "NONE")
    logger.info("Trainable LoRA params: %d", len(lora_trainable))
    if lora_trainable:
        logger.info("Sample LoRA params: %s", lora_trainable[:10])
    logger.info("==================================")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        # LoRA-friendly
        learning_rate=1e-4,
        num_train_epochs=5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,

        # eval/save
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        # speed
        fp16=args.fp16,

        # logs
        logging_steps=50,
        report_to=[],   # or "none"
    )

    trainer = WeightedCELossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        # tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(tokenized_ds["test"])
    pred = trainer.predict(tokenized_ds["test"])
    logits = pred.predictions
    probs = _softmax(logits)
    y_pred = probs.argmax(axis=1)
    y_true = pred.label_ids

    confidence = probs.max(axis=1)
    if probs.shape[1] > 1:
        top2 = np.partition(probs, -2, axis=1)[:, -2]
    else:
        top2 = np.zeros(len(probs), dtype=np.float32)
    margin = confidence - top2
    correct = (y_pred == y_true).astype(int)

    text_col = _resolve_text_column(tokenized_ds["test"].column_names)
    if not text_col:
        logger.warning("No text column found in test set; exporting empty text field.")
        raw_texts = [""] * len(y_pred)
    else:
        raw_texts = tokenized_ds["test"][text_col]

    error_type = np.where(
        correct == 0,
        "misclassification",
        np.where(confidence < LOW_CONFIDENCE_THRESHOLD, "low_confidence", "ok"),
    )

    evaluation_df = pd.DataFrame(
        {
            "run_id": run_id,
            "timestamp": run_timestamp,
            "split": "test",
            "text": [str(t)[:240] for t in raw_texts],
            "label": y_true.astype(int),
            "pred": y_pred.astype(int),
            "correct": correct.astype(int),
            "confidence": confidence.astype(float),
            "margin": margin.astype(float),
            "error_type": error_type,
            "model_name": args.model_name,
            "use_lora": bool(args.use_lora),
            "fp16": bool(args.fp16),
        }
    )

    eval_log_path = os.path.join(args.output_dir, "evaluation_log.csv")
    evaluation_df.to_csv(eval_log_path, index=False)
    logger.info("Evaluation log saved to %s (%d rows)", eval_log_path, len(evaluation_df))

    error_priority = {"misclassification": 0, "low_confidence": 1}
    error_df = (
        evaluation_df[evaluation_df["error_type"] != "ok"]
        .assign(_error_rank=lambda d: d["error_type"].map(error_priority).fillna(2).astype(int))
        .copy()
        .sort_values(
            by=["_error_rank", "confidence", "margin"],
            ascending=[True, True, True],
        )
        .head(ERROR_TOP_K)
        .drop(columns=["_error_rank"])
    )
    error_samples_path = os.path.join(args.output_dir, "error_samples.csv")
    error_df.to_csv(error_samples_path, index=False)
    logger.info("Error samples saved to %s (%d rows)", error_samples_path, len(error_df))
    logger.info("Pred label distribution: %s", Counter(y_pred))

    logger.info("Test metrics: %s", test_metrics)

    # ===== FINAL RESULTS EXPORT =====
    final_results = {
        "test_metrics": test_metrics,
        "pred_label_distribution": dict(Counter(y_pred)),
        "training_config": {
            "learning_rate": training_args.learning_rate,
            "epochs": training_args.num_train_epochs,
            "warmup_ratio": training_args.warmup_ratio,
            "weight_decay": training_args.weight_decay,
            "batch_size": training_args.per_device_train_batch_size,
            "lora_target_modules": ["query", "key", "value"],
            "fp16": training_args.fp16,
            "model_name": args.model_name,
            "use_lora": args.use_lora,
            "run_id": run_id,
            "timestamp": run_timestamp,
            "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
            "error_top_k": ERROR_TOP_K,
        }
    }

    def _json_serializer(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    final_path = os.path.join(args.output_dir, "final_results.json")
    with open(final_path, "w") as f:
        json.dump(final_results, f, indent=2, default=_json_serializer)

    logger.info("Final results saved to %s", final_path)
    # =================================

    logger.info("Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.use_s3:
        # Replace with your actual bucket and prefix
        bucket_name = "your-bucket-name"
        prefix = "financial-nlp/model"
        logger.info("Simulating upload of artifacts to s3://%s/%s", bucket_name, prefix)
        upload_artifacts_to_s3(args.output_dir, bucket=bucket_name, prefix=prefix)


if __name__ == "__main__":
    main()
