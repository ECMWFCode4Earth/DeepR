import logging
from dataclasses import dataclass, field

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from deepr.model.diffusion import SuperResolutionDenoiseDiffusion
from deepr.visualizations import plot_maps

logger = logging.getLogger(__name__)


@dataclass
class DiffusionTrainingConfiguration:
    eps_model: torch.nn.Module
    dataset: torch.utils.data.Dataset
    n_steps: int = 1_000
    batch_size: int = 16
    n_samples: int = 16
    learning_rate: float = 2e-5
    epochs: int = 100
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    device: torch.device = ""
    diffusion: SuperResolutionDenoiseDiffusion = field(init=False)
    data_loader: torch.utils.data.DataLoader = field(init=False)

    def __post_init__(self):
        self.eps_model = self.eps_model.to(self.device)

        self.diffusion = SuperResolutionDenoiseDiffusion(
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

        self.writer = SummaryWriter(log_dir=None)
        self.writer.add_scalar(
            "num_trainable_parameters",
            sum(p.numel() for p in self.eps_model.parameters() if p.requires_grad),
        )

    def sample(self, epoch: int):
        for era5, cerra in self.data_loader:
            output_shape = tuple(cerra.shape[-3:])
            n_samples = min(self.n_samples, self.batch_size)
            coarse_images = era5[:n_samples, ...]
            with torch.no_grad():
                x = torch.randn([n_samples, *output_shape], device=self.device)

                # Remove noise for T steps
                for t_ in range(self.n_steps):
                    t = self.n_steps - t_ - 1
                    x = self.diffusion.p_sample(
                        coarse_images, x, x.new_full((n_samples,), t, dtype=torch.long)
                    )

                if epoch == 0:  # Only first epoch
                    inputs = self.diffusion.merge_net_inputs(coarse_images, x)
                    self.writer.add_graph(
                        self.eps_model, [inputs, torch.ones(inputs.shape[0])]
                    )
                if epoch % 5 == 0:  # Each 5 epochs
                    fig = plot_maps.get_figure_model_samples(
                        coarse_images, cerra[:n_samples, ...], x
                    )
                    self.writer.add_figure("Samples", fig, epoch)
            return None

    def train(self, epoch: int):
        """
        Training function.

        This method iterates over all batches in the dataset, computing the loss of the
        diffusion model for each one and training the parameters of the model. It
        represents an epoch of the training phase.

        Arguments
        ---------
            int: training epoch
        """
        losses = []
        for era5, cerra in self.data_loader:
            era5 = era5.to(self.device)
            cerra = cerra.to(self.device)
            self.optimizer.zero_grad()
            loss = self.diffusion.loss(era5, cerra)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()
        self.writer.add_scalar("loss", sum(losses) / len(losses), epoch)

    def run(self):
        logger.info(
            f"STARTING: SuperResolution Diffusion model is training ... "
            f"({self.epochs} epochs)."
        )
        for epoch in trange(self.epochs):
            self.train(epoch)
            self.sample(epoch)
            self.writer.flush()
        self.writer.close()
        logger.info("FINISHED: The SuperResolution Diffusion model is trained.")
