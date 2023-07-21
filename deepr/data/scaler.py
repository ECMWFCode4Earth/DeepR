import os
import pickle
from typing import Tuple

import pandas
import torch
import xarray

from deepr.data.files import DataFileCollection


class XarrayStandardScaler:
    def __init__(
        self, scaling_files: DataFileCollection, scaling_method: str, cache_file: str
    ):
        """
        Initialize the XarrayStandardScaler object.

        Parameters
        ----------
        scaling_files : DataFileCollection
            Data files from which the XarrayStandardScaler wants to be calculated.
        scaling_method: str
            Method to perform the scaling (pixel-wise, domain-wise, ...)
        cache_file : str
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
            return (
                mean.mean("longitude").mean("latitude"),
                std.mean("longitude").mean("latitude"),
            )
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
        data_indexes = []
        for index in range(data.shape[0]):
            data_index = data[index, :, :, :]
            std_index = (
                self.standard_deviation.sel(month=month[index], method="nearest")
                .to_array()
                .values
            )
            mean_index = (
                self.average.sel(month=month[index], method="nearest").to_array().values
            )
            data_index_inverse = (data_index * std_index) + mean_index
            data_indexes.append(data_index_inverse.unsqueeze(1))
        data_inverse = torch.cat(data_indexes, dim=0)
        return data_inverse

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
