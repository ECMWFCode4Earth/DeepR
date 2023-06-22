import os
from pathlib import Path
from typing import Tuple

import pandas
import torch
import xarray
from joblib import Memory

from deepr.data.files import DataFileCollection


class XarrayStandardScaler:
    def __init__(self, files: DataFileCollection, cache_directory: Path):
        """
        Initialize the XarrayStandardScaler object.

        Parameters
        ----------
        files : DataFileCollection
            Data files from which the XarrayStandardScaler wants to be calculated.
        cache_directory : str
            Directory path to store the cache files.
        """
        self.files = files
        self.cache_directory = cache_directory
        os.makedirs(self.cache_directory, exist_ok=True)
        self.average, self.standard_deviation = self.get_parameters()
        self.average.load()
        self.standard_deviation.load()

    def create_memory(self):
        """
        Create a joblib Memory object with the specified cache directory.

        Returns
        -------
        Memory
            A joblib Memory object with the cache directory.
        """
        memory = Memory(location=str(self.cache_directory), verbose=0)
        return memory

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

        @self.create_memory().cache
        def compute_parameters():
            dataset = xarray.open_mfdataset(
                [file.to_path() for file in self.files.collection]
            )
            mean = dataset.groupby("time.month").mean()
            std = dataset.groupby("time.month").std()
            return mean, std

        return compute_parameters()

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

    def inverse_transform(
        self, data: torch.Tensor, month: torch.Tensor
    ) -> torch.Tensor:
        """Inverse the standard scaling to the input dataset."""
        return data
