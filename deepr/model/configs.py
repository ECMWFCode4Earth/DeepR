import logging
import os
from typing import List, Optional

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
    save_image_epochs: Optional[int] = None
    save_model_epochs: Optional[int] = None
    instance_norm: Optional[bool] = False
    hour_embed_type: str = "none"
    hour_embed_size: int = 64
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

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def _is_last_epoch(self, epoch: int):
        return epoch == self.num_epochs - 1

    def is_save_model_time(self, epoch: int):
        if self.save_model_epochs is None:
            return False or self._is_last_epoch(epoch)

        _epoch_save = (epoch + 1) % self.save_model_epochs == 0
        return _epoch_save or self._is_last_epoch(epoch)

    def is_save_images_time(self, epoch: int):
        if self.save_image_epochs is None:
            return False or self._is_last_epoch(epoch)

        _epoch_save = (epoch + 1) % self.save_image_epochs == 0
        return _epoch_save or self._is_last_epoch(epoch)
