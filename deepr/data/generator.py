import os

import numpy as np
import xarray as xr
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, features_dir, labels_dir, batch_size, shuffle=True):
        """
        Initialize the DataGenerator class.

        Parameters
        ----------
        features_dir : str
            Directory path where the feature files are stored.
        labels_dir : str
            Directory path where the label files are stored.
        batch_size : int
            Number of samples per batch.
        shuffle : bool, optional
            Whether to shuffle the samples, by default True.
        """
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.feature_files = sorted(os.listdir(features_dir))
        self.label_files = sorted(os.listdir(labels_dir))
        self.num_files = len(self.feature_files)
        self.num_samples = self.get_num_samples()
        self.indexes = np.arange(self.num_samples)
        if shuffle:
            np.random.shuffle(self.indexes)

    def get_num_samples(self):
        """
        Calculate the total number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples in the dataset.
        """
        num_samples = 0
        for feature_file in self.feature_files:
            features_ds = xr.open_dataset(os.path.join(self.features_dir, feature_file))
            num_samples += features_ds.dims["time"]
            features_ds.close()
        return num_samples

    def __len__(self):
        """
        Get the number of batches in the dataset.

        Returns
        -------
        int
            Number of batches in the dataset.
        """
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        """
        Generate a batch of data given the batch index.

        Parameters
        ----------
        index : int
            Index of the batch.

        Returns
        -------
        tuple
            Tuple containing the batch of features and labels.
        """
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, self.num_samples)
        batch_indexes = self.indexes[start_index:end_index]
        batch_feature_files, batch_sample_indexes = self.get_files_and_sample_indexes(
            batch_indexes
        )
        batch_features, batch_labels = self.load_data(
            batch_feature_files, batch_sample_indexes
        )
        return batch_features, batch_labels

    def get_files_and_sample_indexes(self, batch_indexes):
        """
        Get the feature files and corresponding sample indexes for a batch.

        Parameters
        ----------
        batch_indexes : list
            List of indexes for the samples in the batch.

        Returns
        -------
        tuple
            Tuple containing the feature files and sample indexes.
        """
        file_indexes = []
        sample_indexes = []
        for index in batch_indexes:
            for file_index, feature_file in enumerate(self.feature_files):
                features_ds = xr.open_dataset(
                    os.path.join(self.features_dir, feature_file)
                )
                num_samples = features_ds.dims["time"]
                if index < num_samples:
                    file_indexes.append(file_index)
                    sample_indexes.append(index)
                    features_ds.close()
                    break
                index -= num_samples
                features_ds.close()
        batch_feature_files = [self.feature_files[i] for i in file_indexes]
        batch_sample_indexes = sample_indexes
        return batch_feature_files, batch_sample_indexes

    def load_data(self, feature_files, sample_indexes):
        """
        Load the data for a batch of samples.

        Parameters
        ----------
        feature_files : list
            List of feature files for the samples in the batch.
        sample_indexes : list
            List of sample indexes in the feature files.

        Returns
        -------
        tuple
            Tuple containing the batch of features and labels.
        """
        batch_features, batch_labels = [], []
        for feature_file, sample_index in zip(feature_files, sample_indexes):
            features_ds = xr.open_dataset(os.path.join(self.features_dir, feature_file))
            labels_ds = xr.open_dataset(
                os.path.join(self.labels_dir, feature_file)
            )  # Assuming same file for features and labels
            features = features_ds["t2m"].isel(time=sample_index).values
            labels = labels_ds["t2m"].isel(time=sample_index).values
            batch_features.append(features)
            batch_labels.append(labels)
            features_ds.close()
            labels_ds.close()
        return np.array(batch_features), np.array(batch_labels)

    def on_epoch_end(self):
        """Perform any necessary actions at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indexes)
