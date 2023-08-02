from pathlib import Path

import xarray

from deepr.utilities.logger import get_logger
from deepr.validation.netcdf.metrics import Metrics
from deepr.validation.netcdf.visualize import Visualization

logger = get_logger("Validation Configuration")


class ValidationConfig:
    def __init__(self, validation_configuration: dict):
        """
        Initialize the ValidationConfig object.

        Parameters
        ----------
        validation_configuration : dict
            A dictionary containing the validation configuration parameters.

        """
        logger.info(
            f"Initializing model configuration object with parameters: "
            f"{validation_configuration}"
        )
        self.validation_configuration = validation_configuration

    def run(self):
        model_predictions, baseline_predictions = self.open_predictions()
        model_predictions.load()
        baseline_predictions.load()

        observations = self.open_observations()
        observations.load()

        self.validate_predictions(observations, model_predictions, baseline_predictions)

    def open_predictions(self):
        if self.validation_configuration["model_predictions_location"] is not None:
            model_predictions = xarray.open_mfdataset(
                f"{self.validation_configuration['model_predictions_location']}/*.nc"
            )
        else:
            model_predictions = None

        if self.validation_configuration["baseline_predictions_location"] is not None:
            baseline_predictions = xarray.open_mfdataset(
                f"{self.validation_configuration['baseline_predictions_location']}/*.nc"
            )
        else:
            baseline_predictions = None
        return model_predictions.sortby("time"), baseline_predictions.sortby("time")

    def open_observations(self):
        if self.validation_configuration["observations_location"] is not None:
            observations = xarray.open_mfdataset(
                f"{self.validation_configuration['observations_location']}/*.nc"
            )
            observations = observations.rename_vars({"t2m": "observation"})
        else:
            observations = None
        return observations.sortby("time")

    def validate_predictions(
        self,
        observations: xarray.Dataset,
        predictions: xarray.Dataset,
        baselines: xarray.Dataset,
    ):
        """
        Validate predictions against observations and benchmarks.

        Parameters
        ----------
        observations : xarray.Dataset
            A dataset containing the observations.
        predictions : xarray.Dataset
            A dataset containing the predictions.
        baselines : xarray.Dataset
            A dataset containing the benchmarks.
        """
        # Metrics datasets for the different sample types
        model_metrics_dataset = Metrics(
            observations=observations.rename_vars({"observation": "variable"}),
            predictions=predictions.rename_vars({"prediction": "variable"}),
            output_directory=Path(
                f'{self.validation_configuration["validation_dir"]}/metrics/model/'
            ),
        ).get_metrics()
        baseline_metrics_dataset = Metrics(
            observations=observations.rename_vars({"observation": "variable"}),
            predictions=baselines.rename_vars({"prediction": "variable"}),
            output_directory=Path(
                f'{self.validation_configuration["validation_dir"]}/metrics/baseline/'
            ),
        ).get_metrics()

        # Visualizations for the different data types
        Visualization(
            observations=observations.rename_vars({"observation": "variable"}),
            predictions=predictions.rename_vars({"prediction": "variable"}),
            baselines=baselines.rename_vars({"prediction": "variable"}),
            model_metrics=model_metrics_dataset,
            baseline_metrics=baseline_metrics_dataset,
            visualization_types=self.validation_configuration["visualization_types"],
            output_directory=Path(
                f'{self.validation_configuration["validation_dir"]}/figures/'
                f'{self.validation_configuration["model_name"]}'
            ),
        ).get_visualizations()

        return model_metrics_dataset, baseline_metrics_dataset


if __name__ == "__main__":
    from deepr.utilities.yml import read_yaml_file

    ValidationConfig(
        read_yaml_file("../../../resources/configuration_validation_netcdf.yml")[
            "validation"
        ]
    ).run()
