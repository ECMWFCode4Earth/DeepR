import torch
import xarray
from torch.utils.data import Dataset


class DataGenerator(Dataset):
    def __init__(self, features_files, label_files):
        """
        Initialize the DataGenerator class.

        Parameters
        ----------
        features_dir : str
            Directory path where the feature files are stored.
        labels_dir : str
            Directory path where the label files are stored.
        """
        self.feature_files = sorted(features_files)
        self.label_files = sorted(label_files)
        self.num_samples = self.get_num_samples()

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
            features_ds = xarray.open_dataset(feature_file)
            num_samples += features_ds.dims["time"]
            features_ds.close()
        return num_samples

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
        Get a sample from the dataset given its index.

        Parameters
        ----------
        index : int
            Index of the sample.

        Returns
        -------
        tuple
            Tuple containing the feature and label tensors.
        """
        feature_file, sample_index = self.get_file_and_sample_index(index)
        features, labels = self.load_data(feature_file, sample_index)
        return torch.from_numpy(features), torch.from_numpy(labels)

    def get_file_and_sample_index(self, index):
        """
        Get the feature file and corresponding sample index for a given index.

        Parameters
        ----------
        index : int
            Index of the sample.

        Returns
        -------
        tuple
            Tuple containing the feature file and sample index.
        """
        for file_index, feature_file in enumerate(self.feature_files):
            features_ds = xarray.open_dataset(feature_file)
            num_samples = features_ds.dims["time"]
            if index < num_samples:
                features_ds.close()
                return feature_file, index
            index -= num_samples
            features_ds.close()

    def load_data(self, feature_file, sample_index):
        """
        Load the data for a given sample.

        Parameters
        ----------
        feature_file : str
            Feature file for the sample.
        sample_index : int
            Sample index in the feature file.

        Returns
        -------
        tuple
            Tuple containing the feature and label arrays.
        """
        features_ds = xarray.open_dataset(feature_file)
        labels_ds = xarray.open_dataset(feature_file)
        features = features_ds["t2m"].isel(time=sample_index).values
        labels = labels_ds["t2m"].isel(time=sample_index).values
        features_ds.close()
        labels_ds.close()
        return features, labels
