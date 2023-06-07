import numpy
import pandas
import torch
import xarray
from torch.utils.data import IterableDataset

from deepr.data.configuration import DataFileCollection
from deepr.data.scaler import XarrayStandardScaler


class DataGenerator(IterableDataset):
    def __init__(
        self,
        features_files: DataFileCollection,
        label_files: DataFileCollection,
        features_scaler: XarrayStandardScaler,
        label_scaler: XarrayStandardScaler,
    ):
        """
        Initialize the DataGenerator class.

        Parameters
        ----------
        features_files : DataFileCollection
            Collection of feature DataFile objects.
        label_files : DataFileCollection
            Collection of label DataFile objects.
        features_scaler: XarrayStandardScaler
            Scaler object with which to apply the standardization
        label_scaler: XarrayStandardScaler
            Scaler object with which to apply the standardization
        """
        super(DataGenerator).__init__()
        self.feature_files = features_files
        self.label_files = label_files
        self.features_scaler = features_scaler
        self.label_scaler = label_scaler
        self.number_files = len(self.label_files.collection)
        self.file_index = 0
        self.num_samples = self.get_num_samples()
        self.label_ds = None
        self.features_ds = None

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return self.num_samples

    def get_num_samples(self):
        """
        Calculate the total number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples in the dataset.
        """
        num_samples = 0
        for label_file in self.label_files.collection:
            label_ds = xarray.open_dataset(label_file.to_path())
            num_samples += label_ds.dims["time"]
            label_ds.close()
        return num_samples

    def __iter__(self):
        """
        Retrieve a batch of data given an index.

        Returns
        -------
        tuple
            A tuple containing the batch of feature and label data.

        Raises
        ------
        IndexError
            If the index is out of range.
        """
        if self.label_ds is None and self.features_ds is None:
            self.load_data()

        for time_index, time_value in enumerate(self.label_ds.time.values):
            time_index += 1
            features_ds_batch = self.features_ds.sel(time=time_value)
            if self.features_scaler:
                features_ds_batch = self.features_scaler.apply_scaler(features_ds_batch)
            label_ds_batch = self.label_ds.sel(time=time_value)
            if self.label_scaler:
                label_ds_batch = self.label_scaler.apply_scaler(label_ds_batch)

            time_value = pandas.to_datetime(time_value)
            time_value_batch = numpy.array(
                [time_value.hour, time_value.day, time_value.month, time_value.year]
            )
            batch = (
                torch.as_tensor(features_ds_batch.to_array().to_numpy()),
                torch.as_tensor(label_ds_batch.to_array().to_numpy()),
                torch.as_tensor(time_value_batch),
            )
            if time_index == len(self.label_ds.time.values):
                self.label_ds = None
                self.features_ds = None

            yield batch

    def load_data(self):
        """
        Load the data from the given label file and feature files.

        Returns
        -------
        tuple
            A tuple containing the feature and label datasets.

        Notes
        -----
        This is a static method and does not require an instance of the class.

        The label file and feature files are expected to be in NetCDF format.

        The feature datasets are merged into a single dataset using xarray.merge().
        """
        label_file = self.label_files.collection[self.file_index]
        features_files = self.feature_files.find_data(
            **{"temporal_coverage": label_file.temporal_coverage}
        )

        label_ds = xarray.open_dataset(label_file.to_path())
        self.label_ds = label_ds.sel(
            latitude=slice(
                label_file.spatial_coverage["latitude"][0],
                label_file.spatial_coverage["latitude"][1],
            ),
            longitude=slice(
                label_file.spatial_coverage["longitude"][0],
                label_file.spatial_coverage["longitude"][1],
            ),
        )
        features_datasets = []
        for features_file in features_files.collection:
            features_ds = xarray.open_dataset(features_file.to_path())
            features_ds = features_ds.sel(
                latitude=slice(
                    features_file.spatial_coverage["latitude"][0],
                    features_file.spatial_coverage["latitude"][1],
                ),
                longitude=slice(
                    features_file.spatial_coverage["longitude"][0],
                    features_file.spatial_coverage["longitude"][1],
                ),
            )
            features_datasets.append(features_ds)
        self.features_ds = xarray.merge(features_datasets)

        self.file_index += 1
