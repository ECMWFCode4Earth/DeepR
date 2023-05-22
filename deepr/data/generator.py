import torch
import xarray
from torch.utils.data import Dataset


class DataGenerator(Dataset):
    def __init__(self, features_files, label_files):
        """
        Initialize the DataGenerator class.

        Parameters
        ----------
        features_files : DataFileCollection
            Collection of feature DataFile objects.
        label_files : DataFileCollection
            Collection of label DataFile objects.
        """
        self.feature_files = features_files
        self.label_files = label_files
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

    def __getitem__(self, index):
        """
        Retrieve a batch of data given an index.

        Parameters
        ----------
        index : int
            Index of the batch.

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
            file_idx, _ = self.get_indices(index)
            label_file = self.label_files.collection[file_idx]
            features_files = self.feature_files.find_data(**{"date": label_file.date})
            self.features_ds, self.label_ds = self.load_data(
                label_file=label_file, features_files=features_files
            )

        file_idx, sample_idx = self.get_indices(index)

        features_ds_batch = self.features_ds.isel(time=sample_idx)
        label_ds_batch = self.label_ds.isel(time=sample_idx)

        if sample_idx >= self.label_ds.dims["time"]:
            self.label_ds = None
            self.features_ds = None

        batch = (
            torch.as_tensor(features_ds_batch.to_array().to_numpy()),
            torch.as_tensor(label_ds_batch.to_array().to_numpy()),
        )
        return batch

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

    def get_indices(self, index):
        """
        Calculate the file index and sample index based on the given overall index.

        Parameters
        ----------
        index : int
            Overall index of a sample in the dataset.

        Returns
        -------
        tuple
            A tuple containing the file index and sample index.

        Raises
        ------
        IndexError
            If the index is out of range.
        """
        file_idx = 0
        sample_idx = index
        for label_file in self.label_files.collection:
            num_samples = xarray.open_dataset(label_file.to_path()).dims["time"]
            if sample_idx < num_samples:
                break
            else:
                sample_idx -= num_samples
                file_idx += 1
        return file_idx, sample_idx

    @staticmethod
    def load_data(features_files, label_file):
        """
        Load the data from the given label file and feature files.

        Parameters
        ----------
        features_files : DataFileCollection
            The DataFileCollection object containing the feature files.
        label_file : DataFile
            The DataFile object representing the label file.

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
        label_ds = xarray.open_dataset(label_file.to_path())
        features_datasets = []
        for features_file in features_files.collection:
            features_ds = xarray.open_dataset(features_file.to_path())
            features_datasets.append(features_ds)
        features_ds = xarray.merge(features_datasets)
        return features_ds, label_ds
