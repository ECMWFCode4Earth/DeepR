import logging

from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    train_batch_size: int = 16
    val_batch_size: int = 16  # how many images to sample during evaluation
    validation_split: float = 0.2
    num_epochs: int = 50
    num_samples: int = 3
    gradient_accumulation_steps = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 10
    save_model_epochs: int = 30
    device: str = "cpu"
    mixed_precision: str = (
        "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    )
    output_dir: str = "ddpm-probando-128"  # the model name locally and on the HF Hub
    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False
    overwrite_output_dir: bool = (
        True  # overwrite the old model when re-running the notebook
    )
    seed: int = 0
