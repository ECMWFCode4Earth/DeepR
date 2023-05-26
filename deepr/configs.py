import datetime
from dataclasses import dataclass, field
from typing import Dict, List

import pandas
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from deepr.data.files import DataFile, DataFileCollection
from deepr.model.diffusion import DenoiseDiffusion


class DataConfiguration:
    def __init__(self, data_configuration: Dict):
        """
        Initialize the DataConfiguration class.

        Parameters
        ----------
        data_configuration : dict
            Data configuration dictionary containing features_configuration, label_configuration,
            and common_configuration.
        """
        self.features_configuration = data_configuration["features_configuration"]
        self.label_configuration = data_configuration["label_configuration"]
        self.common_configuration = data_configuration["common_configuration"]

    def get_dates(self) -> List:
        """
        Get the dates based on the temporal coverage and frequency.

        Returns
        -------
        dates : List
            A list containing the dates within the temporal coverage.
        """
        temporal_coverage = self.common_configuration["temporal_coverage"]
        dates = pandas.date_range(
            start=temporal_coverage["start"],
            end=temporal_coverage["end"],
            freq=temporal_coverage["frequency"],
        )
        dates = [datetime.datetime.strftime(date, "%Y%m") for date in dates]
        return dates

    def get_features(self) -> DataFileCollection:
        """
        Get the list of feature files based on the features_configuration.

        Returns
        -------
        list
            List of DataFile objects representing the feature files.

        Raises
        ------
        FileNotFoundError
            If no file was found for the defined features_configuration.
        """
        # Get the dates for the features
        features_dates = self.get_dates()

        # Initialize the list of features
        features_files = DataFileCollection(collection=[])

        # Loop through each date in the features_configuration
        for features_date in features_dates:
            # Loop through each variable in the features_configuration
            for variable in self.features_configuration["variables"]:
                features_file = DataFile(
                    base_dir=self.features_configuration["data_dir"],
                    variable=variable,
                    dataset=self.features_configuration["data_name"],
                    temporal_coverage=features_date,
                    spatial_resolution=self.features_configuration[
                        "spatial_resolution"
                    ],
                    spatial_coverage=self.features_configuration["spatial_coverage"],
                )
                if features_file.exist():
                    features_files.append_data(features_file)

        if not len(features_files):
            raise FileNotFoundError(
                "No file was found for the defined features_configuration."
            )

        return features_files

    def get_label(self) -> DataFileCollection:
        """
        Get the list of label files based on the label_configuration.

        Returns
        -------
        list
            List of DataFile objects representing the label files.

        Raises
        ------
        FileNotFoundError
            If no file was found for the defined label_configuration.
        """
        # Get the dates for the labels
        label_dates = self.get_dates()

        # Initialize the list of labels
        label_files = DataFileCollection(collection=[])

        # Loop through each date in the label_configuration
        for label_date in label_dates:
            label_file = DataFile(
                base_dir=self.label_configuration["data_dir"],
                variable=self.label_configuration["variable"],
                dataset=self.label_configuration["data_name"],
                temporal_coverage=label_date,
                spatial_resolution=self.label_configuration["spatial_resolution"],
                spatial_coverage=self.label_configuration["spatial_coverage"],
            )
            if label_file.exist():
                label_files.append_data(label_file)

        if not len(label_files):
            raise FileNotFoundError(
                "No file was found for the defined label_configuration."
            )

        return label_files


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
