import logging
import os
from typing import Dict, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_from_s3(bucket: str, key: str, local_path: str = "/tmp/data.csv") -> str:
    """
    Placeholder to download a file from S3 to a local path.
    """
    s3_client = boto3.client("s3")
    logger.info("Downloading %s from bucket %s to %s", key, bucket, local_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket, key, local_path)
    return local_path


def load_financial_phrasebank(
    subset: str = "sentences_allagree",
    use_s3: bool = False,
    bucket: Optional[str] = None,
    key: Optional[str] = None,
    local_path: str = "/tmp/financial_phrasebank.csv",
) -> Dataset:
    """
    Load the financial_phrasebank dataset. If use_s3 is True, download from S3;
    otherwise, load from HuggingFace Datasets.
    """
    if use_s3:
        if not bucket or not key:
            raise ValueError("bucket and key must be provided when use_s3=True.")
        local_file = load_from_s3(bucket, key, local_path)
        df = pd.read_csv(local_file)
        return Dataset.from_pandas(df)
    logger.info("Loading financial_phrasebank (%s) from HuggingFace hub.", subset)
    return load_dataset("financial_phrasebank", subset)["train"]


def stratified_splits(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified train/val/test split on a DataFrame.
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size + val_size,
        random_state=seed,
        stratify=df[label_col],
    )
    relative_val_size = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_val_size,
        random_state=seed,
        stratify=temp_df[label_col],
    )
    return train_df, val_df, test_df


def tokenize_datasets(
    datasets: Dict[str, Dataset],
    tokenizer,
    text_col: str = "sentence",
    max_length: int = 128,
) -> DatasetDict:
    """
    Tokenize datasets using the provided tokenizer.
    """
    def _tokenize(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = {}
    for split_name, split_ds in datasets.items():
        tokenized[split_name] = split_ds.map(_tokenize, batched=True)
        tokenized[split_name] = tokenized[split_name].rename_column("label", "labels")
        tokenized[split_name].set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
            output_all_columns=True,
        )
    return DatasetDict(tokenized)


def compute_class_weights(labels: pd.Series) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalance handling.
    """
    counts = labels.value_counts().sort_index()
    num_classes = len(counts)
    total = counts.sum()
    weights = {cls: total / (num_classes * count) for cls, count in counts.items()}
    weight_list = [weights[i] for i in range(num_classes)]
    return torch.tensor(weight_list, dtype=torch.float)


def prepare_data(
    tokenizer,
    subset: str = "sentences_allagree",
    use_s3: bool = False,
    bucket: Optional[str] = None,
    key: Optional[str] = None,
    max_length: int = 128,
    seed: int = 42,
) -> Tuple[DatasetDict, torch.Tensor]:
    """
    End-to-end data preparation:
      1) Load dataset (HF or S3)
      2) Stratified split
      3) Tokenize
      4) Compute class weights from training labels
    Returns tokenized DatasetDict and class_weights tensor.
    """
    raw_ds = load_financial_phrasebank(
        subset=subset,
        use_s3=use_s3,
        bucket=bucket,
        key=key,
    )
    df = raw_ds.to_pandas()
    train_df, val_df, test_df = stratified_splits(df, label_col="label", seed=seed)

    class_weights = compute_class_weights(train_df["label"])

    datasets = {
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False),
        "test": Dataset.from_pandas(test_df, preserve_index=False),
    }
    tokenized = tokenize_datasets(datasets, tokenizer, text_col="sentence", max_length=max_length)
    logger.info("Prepared datasets with class weights: %s", class_weights.tolist())
    return tokenized, class_weights
