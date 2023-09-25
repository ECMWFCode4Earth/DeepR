import os
import pathlib
import pickle
from typing import Tuple

import numpy as np
import pandas
import torch
import xarray

from deepr.data.files import DataFileCollection


class XarrayStandardScaler:
    def __init__(
        self,
        scaling_files: DataFileCollection,
        scaling_method: str,
        cache_file: pathlib.Path,
    ):
        """
        Initialize the XarrayStandardScaler object.

        Parameters
        ----------
        scaling_files : DataFileCollection
            Data files from which the XarrayStandardScaler wants to be calculated.
        scaling_method: str
            Method to perform the scaling (pixel-wise, domain-wise, ...)
        cache_file : pathlib.Path
            Path to store the pickle file.
        """
        self.scaling_files = scaling_files
        self.scaling_method = scaling_method
        self.cache_file = cache_file

        if os.path.exists(self.cache_file):
            self.load()
        else:
            self.average, self.standard_deviation = self.get_parameters()
            self.average.load()
            self.standard_deviation.load()
            self.save()

    def to_dict(self) -> str:
        average = self.average[list(self.average.data_vars)[0]].values
        std = self.standard_deviation[list(self.standard_deviation.data_vars)[0]].values

        if len(average) == 1:
            average = float(average)
        elif average.ndim in [1, 2]:
            average = average.tolist()
        else:
            raise NotImplementedError("Average of scaler must be 1D or 2D.")

        if len(std) == 1:
            std = float(std)
        elif std.ndim in [1, 2]:
            std = std.tolist()
        else:
            raise NotImplementedError("Standard deviation of scaler must be 1D or 2D.")

        return {
            "method": self.scaling_method,
            "time-agg": "monthly",
            "average": average,
            "standard_deviation": std,
        }

    def get_parameters(self) -> Tuple[xarray.Dataset, xarray.Dataset]:
        """
        Calculate the mean and standard deviation of the dataset parameters.

        Returns
        -------
        mean : xarray.Dataset
            The dataset containing the mean values of the parameters.
        std : xarray.Dataset
            The dataset containing the standard deviation values of the parameters.
        """
        datasets = []
        for file in self.scaling_files.collection:
            file_path = file.to_path()
            dataset = xarray.open_dataset(file_path, chunks=16)
            dataset = dataset.sel(
                latitude=slice(
                    file.spatial_coverage["latitude"][0],
                    file.spatial_coverage["latitude"][1],
                ),
                longitude=slice(
                    file.spatial_coverage["longitude"][0],
                    file.spatial_coverage["longitude"][1],
                ),
            )
            datasets.append(dataset)
        dataset = xarray.concat(datasets, dim="time")
        mean = dataset.groupby("time.month").mean()
        std = dataset.groupby("time.month").std()
        if self.scaling_method == "pixel-wise":
            return mean, std
        elif self.scaling_method == "domain-wise":
            spatial_dims = ["longitude", "latitude"]
            return mean.mean(spatial_dims), std.mean(spatial_dims)
        elif self.scaling_method == "landmask-wise":
            return mean, std
        else:
            raise NotImplementedError

    def apply_scaler(self, ds: xarray.Dataset) -> xarray.Dataset:
        """
        Apply the standard scaling to the input dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to be scaled.

        Returns
        -------
        xarray.Dataset
            Scaled dataset with the same dimensions as the input dataset.

        Notes
        -----
        This method subtracts the average dataset from the input dataset and divides
        it by the standard deviation dataset.
        """
        time_month = pandas.to_datetime(ds.time.values).month
        ds_scaled = ds - self.average.sel(month=time_month, method="nearest")
        ds_scaled = ds_scaled / self.standard_deviation.sel(
            month=time_month, method="nearest"
        )
        return ds_scaled

    def apply_inverse_scaler(
        self, data: torch.Tensor, month: torch.Tensor
    ) -> torch.Tensor:
        """
        Inverse the standard scaling to the input dataset.

        Parameters
        ----------
        data : torch.Tensor
            The input dataset that was previously scaled.
        month : torch.Tensor
            The month tensor used for selecting scaling parameters.

        Returns
        -------
        torch.Tensor
            The dataset with standard scaling inverted.

        """
        std_tensor = self.standard_deviation.sel(
            month=month, method="nearest"
        ).to_array()
        std_tensor = torch.from_numpy(
            std_tensor.squeeze().values[..., np.newaxis, np.newaxis, np.newaxis]
        )
        mean_tensor = self.average.sel(month=month, method="nearest").to_array()
        mean_tensor = torch.from_numpy(
            mean_tensor.squeeze().values[..., np.newaxis, np.newaxis, np.newaxis]
        )

        return data * std_tensor + mean_tensor

    def load(self):
        """Load an XarrayStandardScaler object from a pickle file."""
        with open(self.cache_file, "rb") as f:
            scaler = pickle.load(f)
        self.average = scaler.average
        self.standard_deviation = scaler.standard_deviation

    def save(self):
        """Save the XarrayStandardScaler object to a pickle file."""
        with open(self.cache_file, "wb") as f:
            pickle.dump(self, f)
