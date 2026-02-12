import logging
from typing import Union

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, PreTrainedModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_model(
    model_name: str,
    num_labels: int,
    use_lora: bool = True,
) -> Union[PreTrainedModel, object]:
    """
    Load a base sequence classification model; optionally wrap it with LoRA/PEFT.
    """
    logger.info("Loading base model: %s", model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    if not use_lora:
        logger.info("Returning standard model (no LoRA).")
        return base_model

    logger.info("Applying LoRA (PEFT) to the base model.")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
        bias="none",
    )

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model
