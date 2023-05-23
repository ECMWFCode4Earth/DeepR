from typing import Tuple

import xarray

from deepr.data.files import DataFileCollection


class XarrayStandardScaler:
    def __init__(self, files: DataFileCollection):
        """
        Initialize the XarrayStandardScaler object.

        Parameters
        ----------
        files : DataFileCollection
            Data files from which the XarrayStandardScaler wants to be calculated.
        """
        self.files = files
        self.average, self.standard_deviation = self.get_parameters()

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
        dataset = xarray.open_mfdataset(
            [file.to_path() for file in self.files.collection]
        )
        mean = dataset.mean(dim="time")
        std = dataset.std(dim="time")
        return mean, std

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
        ds_scaled = ds - self.average
        ds_scaled = ds_scaled / self.standard_deviation
        return ds_scaled
