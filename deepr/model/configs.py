from typing import List

import torch
import torch.utils.data
from labml import experiment, monit, tracker
from labml.configs import BaseConfigs

from deepr.model.ddpm import DenoiseDiffusion
from deepr.model.unet import UNet


class Configs(BaseConfigs):
    device: torch.device
    eps_model: UNet
    diffusion: DenoiseDiffusion
    image_channels: int = 1
    image_size: int = 32
    n_channels: int = 12
    n_blocks: int = 2
    channel_multipliers: List[int] = [1, 2, 2, 4]
    is_attention: List[int] = [False, False, False, True]
    n_steps: int = 1_000
    batch_size: int = 16
    n_samples: int = 16
    learning_rate: float = 2e-5
    epochs: int = 1_000
    dataset: torch.utils.data.Dataset
    data_loader: torch.utils.data.DataLoader
    optimizer: torch.optim.Adam

    def init(self):
        self.eps_model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_mults=self.channel_multipliers,
            is_attn=self.is_attention,
            n_blocks=self.n_blocks
        ).to(self.device)

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

        tracker.set_image("sample", True)

    def sample(self):
        with torch.no_grad():
            x = torch.randn(
                [self.n_samples, self.image_channels, self.image_size, self.image_size],
                device=self.device,
            )

            # Remove noise for T steps
            for t_ in monit.iterate("Sample", self.n_steps):
                t = self.n_steps - t_ - 1
                x = self.diffusion.p_sample(
                    x, x.new_full((self.n_samples,), t, dtype=torch.long)
                )

            tracker.save("sample", x)

    def train(self):
        for era5, cerra in monit.iterate("Train", self.data_loader):
            tracker.add_global_step()
            era5 = era5.to(self.device)
            cerra = cerra.to(self.device)
            self.optimizer.zero_grad()
            loss = self.diffusion.loss(cerra)
            loss.backward()
            self.optimizer.step()
            tracker.save("loss", loss)

    def run(self):
        for _ in monit.loop(self.epochs):
            self.train()
            self.sample()
            tracker.new_line()
            experiment.save_checkpoint()
