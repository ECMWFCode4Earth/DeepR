import datetime
from dataclasses import dataclass, field
from typing import Dict, List

import pandas
import torch
from torch.utils.tensorboard import SummaryWriter
from deepr.model.diffusion import DenoiseDiffusion


@dataclass
class DiffusionTrainingConfiguration:
    eps_model: torch.nn.Module
    dataset: torch.utils.data.Dataset
    image_size: int = 32
    n_steps: int = 1_000
    batch_size: int = 16
    n_samples: int = 16
    learning_rate: float = 2e-5
    epochs: int = 1_000
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    device: torch.device = ""
    diffusion: DenoiseDiffusion = field(init=False)
    data_loader: torch.utils.data.DataLoader = field(init=False)

    def __post_init__(self):
        self.eps_model = self.eps_model.to(self.device)

        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            device=self.device,
        )

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, self.batch_size, shuffle=True, pin_memory=True
        )
        self.optimizer = torch.optim.Adam(
            self.eps_model.parameters(), lr=self.learning_rate
        )

        SummaryWriter(log_dir=None)

    def sample(self):
        with torch.no_grad():
            x = torch.randn(
                [self.n_samples, self.image_channels, self.image_size, self.image_size],
                device=self.device,
            )

            # Remove noise for T steps
            for t_ in range(self.n_steps):
                t = self.n_steps - t_ - 1
                x = self.diffusion.p_sample(
                    x, x.new_full((self.n_samples,), t, dtype=torch.long)
                )

            self.writer.add_image("samples", x.cpu())

    def train(self):
        for era5, cerra in self.data_loader:
            era5 = era5.to(self.device)
            cerra = cerra.to(self.device)
            inputs = torch.cat((cerra, era5), dim=1)
            self.optimizer.zero_grad()
            loss = self.diffusion.loss(inputs)
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar("loss", loss.item())

    def run(self):
        for epoch in range(self.epochs):
            self.train()
            self.sample()
            self.writer.flush()
        self.writer.close()
