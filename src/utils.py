import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compute accuracy, macro F1, and weighted F1 for HuggingFace Trainer.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
    logger.debug("Computed metrics: %s", metrics)
    return metrics
