import random
from pathlib import Path
from typing import Union

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr


class Location:
    """
    Represents a location with a name, latitude, and longitude.

    Parameters
    ----------
    name : str
        The name of the location.
    lat : float
        The latitude coordinate of the location.
    lon : float
        The longitude coordinate of the location.
    """

    def __init__(self, name: str, lat: float, lon: float):
        self.name = name
        self.lat = lat
        self.lon = lon

    def __str__(self):
        """
        Return a string representation of the location.

        Returns
        -------
        str
            The string representation of the location in the format "name_lat_lon".
            The latitude and longitude values are converted to strings and any decimal
            points are replaced with hyphens.
        """
        str_lat = str(self.lat).replace(".", "-")
        str_lon = str(self.lon).replace(".", "-")
        return f"{self.name}_{str_lat}_{str_lon}"


class Visualization:
    def __init__(
        self,
        predictions: xr.Dataset,
        observations: xr.Dataset,
        baselines: xr.Dataset,
        model_metrics: xr.Dataset,
        baseline_metrics: xr.Dataset,
        visualization_types: dict,
        output_directory: Path,
    ):
        self.predictions, self.observations, self.baselines = (
            predictions,
            observations,
            baselines,
        )
        self.model_metrics, self.baseline_metrics = (
            model_metrics,
            baseline_metrics,
        )
        self.output_directory = output_directory
        self.visualization_types = visualization_types

    def get_visualizations(self):
        """
        Generate visualizations based on the specified visualization types.

        Returns
        -------
        None
        """
        visualization_dict_to_funct = {
            "metrics_global_map": self.visualize_metrics_global_maps,
            "sample_observation_vs_prediction": self.visualize_sample_observation_vs_prediction,
            "seasonal_scatter_plot_for_a_single_site": self.visualize_seasonal_scatter_plot_for_a_single_site,
            "time_series_plot_for_a_single_site": self.visualize_time_series_plot_for_a_single_site,
            "boxplot_for_a_single_site": self.visualize_boxplot_for_a_single_site,
        }
        for (
            visualization_type_name,
            visualization_type_arg,
        ) in self.visualization_types.items():
            visualization_funct = visualization_dict_to_funct[visualization_type_name]
            visualization_funct(visualization_type_arg)

    def get_output_path(
        self,
        visualization_type: str,
        param: Union[Location, int, str] = None,
    ):
        """
        Retrieve the output path of the different plots, given a location or global map.

        Parameters
        ----------
        visualization_type : str
            Type of visualization.
        param : Union[Location, int, str], optional
            Location or parameter for the visualization. Defaults to None.

        Returns
        -------
        Path
            Output path for the plot.
        """
        param_str = str(param) if param is not None else "global"
        output_path = self.output_directory / f"{visualization_type}_{param_str}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
        return output_path

    def visualize_metrics_global_maps(self, show_baseline: bool):
        """
        Visualize every single metric in the self.model_metrics.

        Parameters
        ----------
        show_baseline : bool
            Flag indicating whether to include baseline metrics in the visualization.

        Returns
        -------
        None
        """
        if not show_baseline:
            for metric_var in list(self.model_metrics.data_vars):
                fig, ax = plt.subplots(
                    figsize=(15, 10),
                    ncols=1,
                    nrows=1,
                    subplot_kw={"projection": ccrs.PlateCarree()},
                )
                ax.coastlines()
                self.model_metrics[metric_var].plot(ax=ax, x="x", y="y")
                ax.set_title(f"Metric {metric_var} visualization")
                ax.set_label(f"{metric_var}")
                plt.tight_layout()
                output_path = self.get_output_path("metrics_global_maps", metric_var)
                plt.savefig(output_path)
                plt.close()
        else:
            for metric_var in list(self.model_metrics.data_vars):
                minvalue, maxvalue = self.get_minofmins_maxofmaxs(
                    self.model_metrics, self.baseline_metrics, metric_var
                )
                fig, axes = plt.subplots(
                    figsize=(15, 10),
                    ncols=1,
                    nrows=2,
                    subplot_kw={"projection": ccrs.PlateCarree()},
                )
                ax1, ax2 = axes
                self.model_metrics[metric_var].plot(
                    ax=ax1,
                    x="longitude",
                    y="latitude",
                    vmin=minvalue,
                    vmax=maxvalue,
                    cbar_kwargs={
                        "ticks": [
                            round(x, 2) for x in np.linspace(minvalue, maxvalue, 7)
                        ]
                    },
                )
                ax1.coastlines()
                ax1.set_title(
                    f"Metric {metric_var} visualization for the model. "
                    f"Mean: {round(float(self.model_metrics[metric_var].mean()), 2)}"
                )
                ax1.set_label(f"{metric_var}")
                self.baseline_metrics[metric_var].plot(
                    ax=ax2,
                    x="longitude",
                    y="latitude",
                    vmin=minvalue,
                    vmax=maxvalue,
                    cbar_kwargs={
                        "ticks": [
                            round(x, 2) for x in np.linspace(minvalue, maxvalue, 7)
                        ],
                        "spacing": "proportional",
                    },
                )
                ax2.coastlines()
                ax2.set_title(
                    f"Metric {metric_var} visualization for the baseline. "
                    f"Mean: {round(float(self.baseline_metrics[metric_var].mean()), 2)}"
                )
                ax2.set_label(f"{metric_var}")
                plt.tight_layout()
                output_path = self.get_output_path("metrics_global_maps", metric_var)
                plt.savefig(output_path)
                plt.close()

    def visualize_sample_observation_vs_prediction(self, number_of_samples: int):
        """
        Visualize the prediction vs observation for a given number of samples.

        Parameters
        ----------
        number_of_samples : int
            Number of samples to visualize.

        Returns
        -------
        None
        """
        samples_done = []
        for i in range(number_of_samples):
            random_sample = random.randint(0, len(self.predictions.time.values))
            while random_sample in samples_done:
                random_sample = random.randint(0, len(self.predictions.time.values))
            samples_done.append(random_sample)
            prediction = self.predictions.isel(time=random_sample)
            observation = self.observations.isel(time=random_sample)
            min_of_mins, max_of_maxs = self.get_minofmins_maxofmaxs(
                prediction, observation, "variable"
            )

            fig, axs = plt.subplots(
                ncols=1, nrows=2, subplot_kw={"projection": ccrs.PlateCarree()}
            )
            plt.suptitle("Prediction VS Observation for one sample")
            ax1, ax2 = axs
            ax1.coastlines()
            prediction.variable.plot(
                ax=ax1, x="longitude", y="latitude", vmin=min_of_mins, vmax=max_of_maxs
            )
            ax1.set_title("Prediction")
            ax2 = fig.add_subplot(212, projection=ccrs.PlateCarree())
            ax2.coastlines()
            observation.variable.plot(
                ax=ax2, x="longitude", y="latitude", vmin=min_of_mins, vmax=max_of_maxs
            )
            ax2.set_title("Observation")
            plt.tight_layout()
            output_path = self.get_output_path("sample_observation_vs_prediction", i)
            plt.savefig(output_path)
            plt.close()

    def visualize_seasonal_scatter_plot_for_a_single_site(self, locations: list):
        """
        Create a seasonal scatter plot for a single site.

        Parameters
        ----------
        locations : list
            List of locations to visualize.

        Returns
        -------
        None
        """
        data_merged = xr.merge(
            [
                self.observations.rename_vars({"variable": "obs"}),
                self.predictions.rename_vars({"variable": "preds"}),
            ]
        )
        for location in locations:
            location_obj = Location(**location)
            location_data = data_merged.sel(
                longitude=location_obj.lon, latitude=location_obj.lat, method="nearest"
            )
            max_of_maxs = location_data.max().to_array().values.max()
            min_of_mins = location_data.min().to_array().values.min()
            # Points for the scatter plot
            seasons = ["DJF", "JJA", "MAM", "SON"]
            for season in seasons:
                site_data_for_season = location_data.sel(
                    time=location_data.time.dt.season.isin([season])
                )
                plt.scatter(
                    site_data_for_season["obs"].values,
                    site_data_for_season["preds"].values,
                    label=season,
                )
            # Axis limits for the scatter plot
            plt.axis([min_of_mins, max_of_maxs, min_of_mins, max_of_maxs])
            # Diagonal line for the scatter plot
            plt.plot(
                list(range(int(min_of_mins) - 2, int(max_of_maxs) + 2)),
                list(range(int(min_of_mins) - 2, int(max_of_maxs) + 2)),
                color="black",
            )
            plt.ylabel("Predictions")
            plt.xlabel("Observations")
            plt.title(f"Scatter plot for test samples at {location['name']}")
            plt.legend()
            output_path = self.get_output_path(
                "seasonal_scatter_plot_for_a_single_site",
                location_obj,
            )
            plt.savefig(output_path)
            plt.close()

    def visualize_time_series_plot_for_a_single_site(
        self,
        locations: list,
        temporal_subset=None,
        color_palette=None,
        aggregate_to_daily=False,
    ):
        """
        Create a time series plot for a single site.

        Parameters
        ----------
        locations : list
            List of locations to visualize.
        temporal_subset : slice or tuple of slice, optional
            Temporal subset to plot. If None, plot the entire time series.
            Example: To plot data from '2023-01-01' to '2023-03-31',
                     use slice('2023-01-01', '2023-04-01').
        color_palette : list, optional
            List of three colors for observations, predictions, and baseline lines.
            Default: ['#4c956c', '#fefee3', '#ffc9b9']
        aggregate_to_daily : bool, optional
            If True, aggregate the data from 3-hourly to daily before plotting.

        Returns
        -------
        None
        """
        data_merged = xr.merge(
            [
                self.observations.rename_vars({"variable": "obs"}),
                self.predictions.rename_vars({"variable": "preds"}),
                self.baselines.rename_vars({"variable": "baseline"}),
            ]
        )

        if color_palette is None:
            color_palette = ["#4c956c", "#fefee3", "#ffc9b9"]

        for location in locations:
            location_obj = Location(**location)
            location_data = data_merged.sel(
                longitude=location_obj.lon, latitude=location_obj.lat, method="nearest"
            )

            if temporal_subset:
                location_data = location_data.sel(time=temporal_subset)

            if aggregate_to_daily:
                location_data = location_data.resample(time="1D").mean()

            max_of_maxs = location_data.max().to_array().values.max()
            min_of_mins = location_data.min().to_array().values.min()

            # Create a larger figure
            plt.figure(figsize=(12, 6))  # Adjust the size as needed

            # Points for the scatter plot
            plt.plot(
                location_data.time.values,
                location_data["obs"].values,
                label="Observation",
                color=color_palette[0],
            )
            plt.plot(
                location_data.time.values,
                location_data["preds"].values,
                label="Prediction",
                color=color_palette[1],
            )
            plt.plot(
                location_data.time.values,
                location_data["baseline"].values,
                label="Baseline",
                color=color_palette[2],
            )

            plt.ylim([min_of_mins, max_of_maxs])

            time_cov_str = (
                f"from "
                f"{pd.to_datetime(data_merged.time.values[0]).strftime('%d-%m-%Y')} "
                f"to "
                f"{pd.to_datetime(data_merged.time.values[-1]).strftime('%d-%m-%Y')}"
            )
            plt.title(
                f"Time series at "
                f"{location['name'].replace('_', ' ').replace('-', ' - ')}, "
                f"{time_cov_str}."
            )
            plt.xlabel("Time")  # Add an appropriate x-axis label
            plt.ylabel("Value")  # Add an appropriate y-axis label
            plt.xticks(rotation=45)  # Rotate x-axis labels to avoid overlapping
            plt.legend()
            output_path = self.get_output_path(
                "time_series_plot_for_a_single_site",
                location_obj,
            )
            plt.savefig(output_path)
            plt.close()

    def visualize_boxplot_for_a_single_site(
        self, locations: list, group_by="station", color_palette=None
    ):
        """
        Create a box plot for a single site.

        Parameters
        ----------
        locations : list
            List of locations to visualize.
        group_by : str, optional
            The grouping option for creating boxplots.
            Options: "hour", "month", or "station".
            Default is "station".
        color_palette : list, optional
            List of three colors for the "Data Type" hue in the box plot.
            Default: ['#4c956c', '#fefee3', '#ffc9b9']

        Returns
        -------
        None
        """
        if group_by not in ["hour", "month", "station"]:
            raise ValueError(
                "Invalid group_by option. Choose from 'hour', 'month', or 'station'."
            )

        original_variable = self.observations.variable.attrs["long_name"]
        data_merged = xr.merge(
            [
                self.observations.rename_vars({"variable": "observations"}),
                self.predictions.rename_vars({"variable": "predictions"}),
                self.baselines.rename_vars({"variable": "baseline"}),
            ]
        )

        if color_palette is None:
            color_palette = ["#4c956c", "#fefee3", "#ffc9b9"]

        for location in locations:
            location_obj = Location(**location)
            location_data = data_merged.sel(
                longitude=location_obj.lon, latitude=location_obj.lat, method="nearest"
            )
            location_data_df = location_data.to_dataframe().drop(
                columns=["longitude", "latitude"]
            )

            if group_by == "hour":
                # Group data by hour of the day
                location_data_df["hour"] = (location_data_df.index.hour // 3) * 3
                hourinttostr = {
                    0: "00:00",
                    3: "03:00",
                    6: "06:00",
                    9: "09:00",
                    12: "12:00",
                    15: "15:00",
                    18: "18:00",
                    21: "21:00",
                }
                location_data_df["Hour"] = [
                    hourinttostr[x] for x in location_data_df["hour"].values
                ]
                location_data_df = location_data_df.drop(columns=["time", "hour"])
            elif group_by == "month":
                # Group data by month
                location_data_df["month"] = location_data_df.index.month
                monthinttostr = {
                    1: "January",
                    2: "February",
                    3: "March",
                    4: "April",
                    5: "May",
                    6: "June",
                    7: "July",
                    8: "August",
                    9: "September",
                    10: "October",
                    11: "November",
                    12: "December",
                }
                location_data_df["Month"] = [
                    monthinttostr[x] for x in location_data_df["month"].values
                ]
                location_data_df = location_data_df.drop(columns=["time", "month"])
            else:
                # Group data by season
                location_data_df = self.add_season_column_to_df(location_data_df)
                location_data_df = location_data_df.drop(columns="time")
                seasoninttostr = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"}
                location_data_df["Season"] = [
                    seasoninttostr[x] for x in location_data_df.season.values
                ]

            location_data_df = (
                location_data_df.stack()
                .reset_index()
                .rename(columns={"level_1": "Data Type", 0: original_variable})
            )

            sns.boxplot(
                data=location_data_df,
                x=group_by.capitalize(),
                y=original_variable,
                hue="Data Type",
                palette=color_palette,
            )

            time_cov_str = (
                f"from "
                f"{pd.to_datetime(data_merged.time.values[0]).strftime('%d-%m-%Y')} "
                f"to "
                f"{pd.to_datetime(data_merged.time.values[-1]).strftime('%d-%m-%Y')}"
            )
            plt.title(
                f"Box plot at "
                f"{location['name'].replace('_', ' ').replace('-', ' - ')}, "
                f"{time_cov_str}."
            )
            output_path = self.get_output_path(
                f"boxplot_for_a_single_site_group_by_{group_by}",
                location_obj,
            )
            plt.savefig(output_path)
            plt.close()

    @staticmethod
    def get_minofmins_maxofmaxs(
        first: xr.Dataset, second: xr.Dataset, variable_name: str
    ):
        """
        Calculate the minimum and maximum values for a variable in two datasets.

        Parameters
        ----------
        first : xr.Dataset
            First dataset.
        second : xr.Dataset
            Second dataset.
        variable_name : str
            Name of the variable.

        Returns
        -------
        Tuple
            Minimum of the minimums, maximum of the maximums.
        """
        min_of_predictions = float(first.min()[variable_name].values)
        min_of_observations = float(second.min()[variable_name].values)
        min_of_mins = min([min_of_observations, min_of_predictions])
        max_of_predictions = float(first.max()[variable_name].values)
        max_of_observations = float(second.max()[variable_name].values)
        max_of_maxs = max([max_of_observations, max_of_predictions])
        return min_of_mins, max_of_maxs

    @staticmethod
    def add_season_column_to_df(location_data_df):
        """
        Add a season column to a DataFrame based on the time column.

        Parameters
        ----------
        location_data_df : pd.DataFrame
            A pd.DataFrame containing time information.

        Returns
        -------
        pd.DataFrame
            DataFrame with the added season column.
        """
        season = (location_data_df.time.dt.month - 1) // 3
        season += (location_data_df.time.dt.month == 3) & (
            location_data_df.time.dt.day >= 20
        )
        season += (location_data_df.time.dt.month == 6) & (
            location_data_df.time.dt.day >= 21
        )
        season += (location_data_df.time.dt.month == 9) & (
            location_data_df.time.dt.day >= 23
        )
        season -= 3 * (
            (location_data_df.time.dt.month == 12)
            & (location_data_df.time.dt.day >= 21)
        ).astype(int)
        location_data_df["season"] = season
        return location_data_df
