import logging
from typing import List

import torch
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    num_epochs: int = 50
    batch_size: int = 2
    num_workers: int = 0
    num_samples: int = 3
    gradient_accumulation_steps = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 10
    save_model_epochs: int = 30
    hour_embed_type: str = "none"
    device: str = "cuda"
    mixed_precision: str = (
        "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    )
    output_dir: str = "ddpm-probando-128"  # the model name locally and on the HF Hub
    push_to_hub: bool = False  # whether to upload the saved model to the HF Hub
    hub_private_repo: bool = False
    hf_repo_name: str = ""
    overwrite_output_dir: bool = (
        True  # overwrite the old model when re-running the notebook
    )
    static_covariables: List[str] = None
    seed: int = 0

    def __post_init__(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.info("CUDA device requested but not available :(")
