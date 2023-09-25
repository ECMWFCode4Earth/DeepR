from datetime import datetime
from typing import Dict, List

import pandas
import xarray

from deepr.data.files import DataFile, DataFileCollection

DATA_SPLIT_NAMES = ["train", "validation", "test"]


class DataConfiguration:
    def __init__(self, data_configuration: Dict):
        """
        Initialize the DataConfiguration class.

        Parameters
        ----------
        data_configuration : dict
            Data configuration dictionary containing :
            features_configuration, label_configuration, and split_coverages.
        """
        self.features_configuration = data_configuration.get(
            "features_configuration", None
        )
        self.label_configuration = data_configuration.get("label_configuration", None)
        self.split_coverages = data_configuration.get("split_coverages", None)

    @staticmethod
    def get_dates(temporal_coverage: dict) -> List[str]:
        """
        Get the dates based on the temporal coverage and frequency.

        Parameters
        ----------
        temporal_coverage: dict
            A dictionary indicating the start, end and frequency of the
            temporal coverage.

        Returns
        -------
        dates : List
            A list containing the dates within the temporal coverage.
        """
        dates = pandas.date_range(
            start=temporal_coverage["start"],
            end=temporal_coverage["end"],
            freq=temporal_coverage["frequency"],
        )
        return [datetime.strftime(date, "%Y%m") for date in dates]

    def get_features(
        self,
    ) -> tuple[
        DataFileCollection | None, DataFileCollection | None, DataFileCollection | None
    ]:
        """
        Get the list of feature files based on the features_configuration.

        Returns
        -------
        tuple
            Tuple of DataFile objects representing the feature files.

        Raises
        ------
        FileNotFoundError
            If no file was found for the defined features_configuration.
        """
        if self.features_configuration is None:
            return None, None, None

        data_splits = {}
        for split_name in DATA_SPLIT_NAMES:
            if split_name not in self.split_coverages.keys():
                data_splits[split_name] = DataFileCollection(collection=[])
                continue

            features_dates = self.get_dates(self.split_coverages[split_name])

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
                        spatial_coverage=self.features_configuration[
                            "spatial_coverage"
                        ],
                    )
                    if features_file.exist():
                        features_files.append_data(features_file)

            if not len(features_files):
                raise FileNotFoundError(
                    "No file was found for the defined features_configuration."
                )

            data_splits[split_name] = features_files

        return tuple([data_splits[split] for split in DATA_SPLIT_NAMES])

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

    def get_labels(
        self,
    ) -> tuple[
        DataFileCollection | None, DataFileCollection | None, DataFileCollection | None
    ]:
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

        data_splits = {}
        for split_name in DATA_SPLIT_NAMES:
            # If split is not specified, it is considered to be empty.
            if split_name not in self.split_coverages.keys():
                data_splits[split_name] = DataFileCollection(collection=[])
                continue

            # Get the dates for the labels
            label_dates = self.get_dates(self.split_coverages[split_name])

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

            data_splits[split_name] = label_files

        return tuple([data_splits[split] for split in DATA_SPLIT_NAMES])

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
            lsm = (lsm > 0.5).astype(int)
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

    def get_plain_dict(self) -> dict:
        """
        Prepare and log the data configuration.

        Returns
        -------
        config : Dict
            The prepared data configuration.
        """
        if self.features_configuration["standardization"].get("to_do", False):
            input_sc = self.features_configuration["standardization"]["method"]
        else:
            input_sc = "No"
        if self.label_configuration.get("standardization", {}).get("to_do", False):
            output_sc = self.label_configuration["standardization"]["method"]
        else:
            output_sc = "No"

        input_area = ",".join(
            [
                f"{k}={v}"
                for k, v in self.features_configuration["spatial_coverage"].items()
            ]
        )
        output_area = ",".join(
            [
                f"{k}={v}"
                for k, v in self.label_configuration["spatial_coverage"].items()
            ]
        )

        hparams = {
            "input-dataset": self.features_configuration["data_name"],
            "input-variables": ",".join(self.features_configuration["variables"]),
            "input-area": input_area,
            "input-scaling": input_sc,
            "input-resolution": self.features_configuration["spatial_resolution"],
            "output-dataset": self.label_configuration["data_name"],
            "output-variable": self.label_configuration["variable"],
            "output-area": output_area,
            "output-scaling": output_sc,
            "output-resolution": self.label_configuration["spatial_resolution"],
        }
        for s, period in self.split_coverages.items():
            start_date = period["start"].replace("-", "/")
            end_date = period["end"].replace("-", "/")
            hparams[f"{s}-coverage"] = f"{start_date}-{end_date}"

        return hparams
