from typing import Dict

import pandas

from deepr.data.files import DataFile, DataFileCollection


class DataConfiguration:
    def __init__(self, data_configuration: Dict):
        """
        Initialize the DataConfiguration class.

        Parameters
        ----------
        data_configuration : dict
            Data configuration dictionary containing features_configuration, label_configuration,
            and common_configuration.
        """
        self.features_configuration = data_configuration["features_configuration"]
        self.label_configuration = data_configuration["label_configuration"]
        self.common_configuration = data_configuration["common_configuration"]

    def get_dates(self) -> pandas.DatetimeIndex:
        """
        Get the dates based on the temporal coverage and frequency.

        Returns
        -------
        pandas.DatetimeIndex
            Datetime index containing the dates within the temporal coverage.
        """
        temporal_coverage = self.common_configuration["temporal_coverage"]
        dates = pandas.date_range(
            start=temporal_coverage["start"],
            end=temporal_coverage["end"],
            freq=self.common_configuration["frequency"],
        )
        return dates

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
        # Get the dates for the features
        features_dates = self.get_dates()

        # Initialize the list of features
        features_files = DataFileCollection(collection=[])

        # Loop through each date in the features_configuration
        for features_date in features_dates:
            # Loop through each variable in the features_configuration
            for variable in self.features_configuration:
                features_file = DataFile(
                    base_dir=self.features_configuration["data_dir"],
                    variable=variable,
                    dataset=self.features_configuration["data_name"],
                    date=features_date,
                    resolution=self.features_configuration["spatial_resolution"],
                )
                if features_file.exist():
                    features_files.append_data(features_file)

        if not len(features_files):
            raise FileNotFoundError(
                "No file was found for the defined features_configuration."
            )

        return features_files

    def get_label(self) -> DataFileCollection:
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
        # Get the dates for the labels
        label_dates = self.get_dates()

        # Initialize the list of labels
        label_files = DataFileCollection(collection=[])

        # Loop through each date in the label_configuration
        for label_date in label_dates:
            label_file = DataFile(
                base_dir=self.label_configuration["data_dir"],
                variable=self.label_configuration["variable"],
                dataset=self.label_configuration["data_name"],
                date=label_date,
                resolution=self.label_configuration["spatial_resolution"],
            )
            if label_file.exist():
                label_files.append_data(label_file)

        if not len(label_files):
            raise FileNotFoundError(
                "No file was found for the defined label_configuration."
            )

        return label_files
