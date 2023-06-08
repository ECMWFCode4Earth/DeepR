from deepr.model.configs import TrainingConfig
import torch
from typing import Dict

def train_nn(
    config: TrainingConfig,
    model,
    train_dataset: torch.utils.data.DataLoader,
    test_dataset: torch.utils.data.DataLoader,
    data_config: Dict = None
):
    pass