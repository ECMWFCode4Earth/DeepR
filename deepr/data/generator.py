import xarray

from deepr.data.files import DataFile, DataFileCollection


class DataGenerator:
    def __init__(self, features_files, label_files, batch_size):
        """
        Initialize the DataGenerator class.

        Parameters
        ----------
        features_files : DataFileCollection
            Collection of feature DataFile objects.
        label_files : DataFileCollection
            Collection of label DataFile objects.
        batch_size : int
            Number of samples per batch.
        """
        self.feature_files = features_files
        self.label_files = label_files
        self.batch_size = batch_size
        self.num_samples = self.get_num_samples()
        self.num_batches = self.num_samples // batch_size
        self.label_ds = None
        self.feature_ds = None

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
        self.count = 0
        self.sample_idx = 0
        self.file_idx = 0
        return self

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return self.num_samples

    def __next__(self):
        """
        Iterate over each batch.

        Raises
        ------
        StopIteration
            When all batches are extracted
        """
        if self.count == self.num_batches:
            raise StopIteration

        if self.label_ds is None and self.feature_ds is None:
            label_file = self.label_files.collection[self.file_idx]
            features_files = self.feature_files.find_data(
                kwargs={"date": label_file.date}
            )
            self.features_ds, self.label_ds = self.load_data(
                label_file=label_file, features_files=features_files
            )

        features_ds_batch = self.feature_ds.isel(
            time=slice(self.sample_idx, self.sample_idx + self.batch_size)
        )
        label_ds_batch = self.label_ds.isel(
            time=slice(self.sample_idx, self.sample_idx + self.batch_size)
        )
        self.sample_idx += self.batch_size

        if self.sample_idx + self.batch_size >= self.label_ds.dims["time"]:
            self.sample_idx = 0
            self.label_ds = None
            self.feature_ds = None
            self.file_idx += 1

        self.count += 1

        batch = features_ds_batch, label_ds_batch
        return batch

    @staticmethod
    def load_data(features_files: DataFileCollection, label_file: DataFile):
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
