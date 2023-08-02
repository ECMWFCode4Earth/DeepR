import pathlib

import xarray as xr
import xskillscore as xs


class Metrics:
    def __init__(
        self,
        observations: xr.Dataset,
        predictions: xr.Dataset,
        output_directory: pathlib.Path,
    ):
        """
        Initialize the Metrics object.

        Parameters
        ----------
        observations : xr.Dataset
            The dataset containing the observations.
        predictions : xr.Dataset
            The dataset containing the predictions.
        output_directory : str or pathlib.Path
            The output directory for storing the metrics.

        """
        self.predictions, self.observations = predictions, observations
        self.output_directory = output_directory

    def get_metrics(self):
        """
        Compute metrics for the predictions and observations.

        Returns
        -------
        metrics_ds : xr.Dataset
            The dataset containing the computed metrics.

        Raises
        ------
        NotImplementedError
            If the problem_type is neither "regression" nor "classification".
        """
        metrics_datasets = []

        regression_metrics = {
            "r2": xs.r2,
            "mae": xs.mae,
            "mape": xs.mape,
            "me": xs.me,
            "mse": xs.mse,
            "rmse": xs.rmse,
        }

        for metric_name, metric_function in regression_metrics.items():
            ds_metric = metric_function(self.observations, self.predictions, dim="time")
            ds_metric = ds_metric.rename_vars({"variable": metric_name})
            metrics_datasets.append(ds_metric)

        metrics_ds = xr.merge(metrics_datasets)
        metrics_ds["obs_mean"] = self.observations.mean(dim="time").rename_vars(
            {"variable": "obs_mean"}
        )["obs_mean"]
        metrics_ds["pred_mean"] = self.predictions.mean(dim="time").rename_vars(
            {"variable": "pred_mean"}
        )["pred_mean"]
        metrics_ds["obs_std"] = self.observations.std(dim="time").rename_vars(
            {"variable": "obs_std"}
        )["obs_std"]
        metrics_ds["pred_std"] = self.predictions.std(dim="time").rename_vars(
            {"variable": "pred_std"}
        )["pred_std"]

        return metrics_ds
