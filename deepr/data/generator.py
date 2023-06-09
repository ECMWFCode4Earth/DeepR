import random
from typing import Tuple

import numpy
import numpy as np
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
        shuffle: bool = False,
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
        self.init_date, self.end_date = self.get_dataset_dates()
        self.features_scaler = features_scaler
        self.label_scaler = label_scaler
        self.number_files = len(self.label_files.collection)
        self.file_index = 0
        self.num_samples = self.get_num_samples()
        self.label_ds = None
        self.features_ds = None
        (
            self.input_shape,
            self.input_channels,
            self.aux_shape,
            self.output_shape,
            self.output_channels,
        ) = self.get_shapes()
        self.shuffle = shuffle
        if self.shuffle:
            combined_files = list(
                zip(self.feature_files.collection, self.label_files.collection)
            )
            random.shuffle(combined_files)
            shuffled_files_features, shuffled_files_labels = zip(*combined_files)
            self.feature_files.collection = shuffled_files_features
            self.label_files.collection = shuffled_files_features

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

    def _read_file(
        self, time_value: np.datetime64
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return (
            torch.as_tensor(features_ds_batch.to_array().to_numpy()),
            torch.as_tensor(label_ds_batch.to_array().to_numpy()),
            torch.as_tensor(time_value_batch),
        )

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
        for self.file_index in range(self.number_files):
            if self.label_ds is None and self.features_ds is None:
                self.load_data()

            for time_index, time_value in enumerate(self.label_ds.time.values):
                batch = self._read_file(time_value)
                if time_index == len(self.label_ds.time.values) - 1:
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

    def get_shapes(self) -> tuple:
        """
        Auxiliary method to retrieve shapes of the input and output data.

        Retrieves the shapes of the input, auxiliary, and output data.

        Returns
        -------
            tuple: A tuple containing the input shape, auxiliary shape, and output shape.

        """
        features_sample, label_sample, aux_sample = next(self.__iter__())
        input_shape = tuple(features_sample.shape[1:])
        input_channels = features_sample.shape[0]
        aux_shape = tuple(aux_sample.shape)
        output_shape = tuple(label_sample.shape[1:])
        out_channels = label_sample.shape[0]
        return input_shape, input_channels, aux_shape, output_shape, out_channels

    def get_dataset_dates(self):
        """
        Retrieve the initial and end dates of the dataset.

        Returns
        -------
        tuple
            A tuple containing the initial and end dates of the dataset.

        Raises
        ------
        IndexError
            If the label files collection is empty.
        """
        files_collection = self.label_files.collection
        init_file = files_collection[0]
        end_file = files_collection[-1]
        init_date = init_file.temporal_coverage
        end_date = end_file.temporal_coverage
        return init_date, end_date
