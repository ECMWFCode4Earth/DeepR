from datetime import datetime
from typing import Dict, List

import pandas
import xarray

from deepr.data.files import DataFile, DataFileCollection


class DataConfiguration:
    def __init__(self, data_configuration: Dict):
        """
        Initialize the DataConfiguration class.

        Parameters
        ----------
        data_configuration : dict
            Data configuration dictionary containing :
            features_configuration, label_configuration, and common_configuration.
        """
        self.features_configuration = data_configuration.get(
            "features_configuration", None
        )
        self.label_configuration = data_configuration.get("label_configuration", None)
        self.common_configuration = data_configuration.get("common_configuration", None)

    def _get_data_splits(self):
        """
        Calculate the validation and test split sizes based on given configuration.

        Returns
        -------
        Tuple[float, float]
            A tuple containing the validation and test split sizes.

        Raises
        ------
        None

        Notes
        -----
        The method extracts data splitting information from the `common_configuration`
        attribute, which should be a dictionary containing the following keys:
            - "data_split" : A dictionary with keys "test" and "validation" (optional)
                             specifying the desired proportions for the test and
                             validation sets, respectively.

        The method returns a tuple (val_split_size, test_split_size) where:
            - val_split_size : float
                The calculated validation split size, or 0 if no validation set
                is defined.
            - test_split_size : float
                The calculated test split size, or 0 if no test set is defined.

        If `common_configuration` is not provided, the method returns default split
        sizes (0.0, 0.0). If the test split size is set to 1.0, it means there is no
        training or validation set, only a test set.
        """
        if self.common_configuration is None:
            return 0.0, 0.0

        data_split = self.common_configuration["data_split"]
        test_split_size = data_split.get("test", 0.0)
        val_split_size = (
            data_split.get("validation", 0.0) / (1 - test_split_size)
            if test_split_size != 1
            else 0
        )
        return val_split_size, test_split_size

    def get_dates(self) -> List[str]:
        """
        Get the dates based on the temporal coverage and frequency.

        Returns
        -------
        dates : List
            A list containing the dates within the temporal coverage.
        """
        temporal_coverage = self.common_configuration["temporal_coverage"]
        dates = pandas.date_range(
            start=temporal_coverage["start"],
            end=temporal_coverage["end"],
            freq=temporal_coverage["frequency"],
        )
        return [datetime.strftime(date, "%Y%m") for date in dates]

    def get_features(self) -> DataFileCollection:
        """
        Get the list of feature files based on the features_configuration.

        Returns
        -------
        list
            List of DataFile objects representing the feature files.

        Raises
        ------
        FileNotFoundError
            If no file was found for the defined features_configuration.
        """
        if self.features_configuration is None:
            return None, None, None

        # Get the dates for the features
        features_dates = self.get_dates()

        # Initialize the list of features
        features_files = DataFileCollection(collection=[])

        # Loop through each date in the features_configuration
        for features_date in features_dates:
            # Loop through each variable in the features_configuration
            for variable in self.features_configuration["variables"]:
                features_file = DataFile(
                    base_dir=self.features_configuration["data_location"],
                    variable=variable,
                    dataset=self.features_configuration["data_name"],
                    temporal_coverage=features_date,
                    spatial_resolution=self.features_configuration[
                        "spatial_resolution"
                    ],
                    spatial_coverage=self.features_configuration["spatial_coverage"],
                )
                if features_file.exist():
                    features_files.append_data(features_file)

        if not len(features_files):
            raise FileNotFoundError(
                "No file was found for the defined features_configuration."
            )

        if self.common_configuration is not None:
            val_split, test_split = self._get_data_splits()
            features_coll_train, features_coll_test = features_files.split_data(
                test_split
            )
            features_coll_train, features_coll_val = features_coll_train.split_data(
                val_split
            )
            return features_coll_train, features_coll_val, features_coll_test
        else:
            return features_files, None, None

    def get_static_features(self):
        """
        Get the static features from provided land mask and orography datasets.

        Returns
        -------
        Tuple[xarray.DataArray or None, xarray.DataArray or None]
            A tuple containing the land mask and orography data arrays.

            - lsm : xarray.DataArray or None
                The land mask data array, representing the presence or absence of land
                for each spatial grid point within the defined spatial coverage.
                If the land mask dataset is not provided, returns None.

            - orog : xarray.DataArray or None
                The orography data array, representing the elevation values of
                the terrain for each spatial grid point within the defined
                spatial coverage.
                If the orography dataset is not provided, returns None.
        """
        spatial_coverage = self.features_configuration["spatial_coverage"]
        if self.features_configuration.get("land_mask_location", None) is not None:
            lsm = (
                xarray.open_dataset(self.features_configuration["land_mask_location"])
                .mean("time")
                .sel(
                    latitude=slice(
                        spatial_coverage["latitude"][0],
                        spatial_coverage["latitude"][1],
                    ),
                    longitude=slice(
                        spatial_coverage["longitude"][0],
                        spatial_coverage["longitude"][1],
                    ),
                )
            )
            lsm = (lsm > 0.5).astype(int)
        else:
            lsm = None

        if self.features_configuration.get("orography_location", None):
            orog = (
                xarray.open_dataset(self.features_configuration["orography_location"])
                .mean("time")
                .sel(
                    latitude=slice(
                        spatial_coverage["latitude"][0],
                        spatial_coverage["latitude"][1],
                    ),
                    longitude=slice(
                        spatial_coverage["longitude"][0],
                        spatial_coverage["longitude"][1],
                    ),
                )
            )
        else:
            orog = None
        return lsm, orog

    def get_labels(self) -> DataFileCollection:
        """
        Get the list of label files based on the label_configuration.

        Returns
        -------
        list
            List of DataFile objects representing the label files.

        Raises
        ------
        FileNotFoundError
            If no file was found for the defined label_configuration.
        """
        if self.label_configuration is None:
            return None, None, None

        # Get the dates for the labels
        label_dates = self.get_dates()

        # Initialize the list of labels
        label_files = DataFileCollection(collection=[])

        # Loop through each date in the label_configuration
        for label_date in label_dates:
            label_file = DataFile(
                base_dir=self.label_configuration["data_location"],
                variable=self.label_configuration["variable"],
                dataset=self.label_configuration["data_name"],
                temporal_coverage=label_date,
                spatial_resolution=self.label_configuration["spatial_resolution"],
                spatial_coverage=self.label_configuration["spatial_coverage"],
            )
            if label_file.exist():
                label_files.append_data(label_file)

        if not len(label_files):
            raise FileNotFoundError(
                "No file was found for the defined label_configuration."
            )

        if self.common_configuration is not None:
            val_split, test_split = self._get_data_splits()
            label_coll_train, label_coll_test = label_files.split_data(test_split)
            label_coll_train, label_coll_val = label_coll_train.split_data(val_split)
            return label_coll_train, label_coll_val, label_coll_test
        else:
            return label_files, None, None

    def get_static_label(self):
        """
        Get the static label from provided land mask and orography datasets.

        Returns
        -------
        Tuple[xarray.DataArray or None, xarray.DataArray or None]
            A tuple containing the land mask and orography data arrays.

            - lsm : xarray.DataArray or None
                The land mask data array, representing the presence or absence of land
                for each spatial grid point within the defined spatial coverage.
                If the land mask dataset is not provided, returns None.

            - orog : xarray.DataArray or None
                The orography data array, representing the elevation values of
                the terrain for each spatial grid point within the defined
                spatial coverage.
                If the orography dataset is not provided, returns None.
        """
        spatial_coverage = self.label_configuration["spatial_coverage"]
        if self.label_configuration.get("land_mask_location", None) is not None:
            lsm = (
                xarray.open_dataset(self.label_configuration["land_mask_location"])
                .mean("time")
                .sel(
                    latitude=slice(
                        spatial_coverage["latitude"][0],
                        spatial_coverage["latitude"][1],
                    ),
                    longitude=slice(
                        spatial_coverage["longitude"][0],
                        spatial_coverage["longitude"][1],
                    ),
                )
            )
        else:
            lsm = None

        if self.label_configuration.get("orography_location", None) is not None:
            orog = (
                xarray.open_dataset(self.label_configuration["orography_location"])
                .mean("time")
                .sel(
                    latitude=slice(
                        spatial_coverage["latitude"][0],
                        spatial_coverage["latitude"][1],
                    ),
                    longitude=slice(
                        spatial_coverage["longitude"][0],
                        spatial_coverage["longitude"][1],
                    ),
                )
            )
        else:
            orog = None
        return lsm, orog
