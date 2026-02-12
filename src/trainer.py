import logging
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from transformers import Trainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class WeightedCELossTrainer(Trainer):
    """
    Custom Trainer using weighted Cross-Entropy loss for class-imbalanced data.
    Pass a torch.Tensor of class weights at initialization.
    """

    def __init__(self, class_weights: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if class_weights is None:
            raise ValueError("class_weights must be provided for WeightedCELossTrainer.")
        # Store as float tensor; will be moved to the correct device in compute_loss
        self.class_weights = class_weights.float()
        logger.info("Initialized WeightedCELossTrainer with class weights: %s", self.class_weights)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Override to apply weighted CrossEntropyLoss.
        Compatible with newer Transformers that pass extra kwargs (e.g., num_items_in_batch).
        """
        labels = inputs.get("labels")

        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits

        weight = self.class_weights.to(logits.device)

        loss_fct = nn.CrossEntropyLoss(weight=weight)

        loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss
    