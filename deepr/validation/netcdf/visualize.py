import random
from pathlib import Path
from typing import Union

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from scipy.stats import norm


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
            The string representation of the location in the format "name".
            The latitude and longitude values are converted to strings and any decimal
            points are replaced with hyphens.
        """
        return f"{self.name.lower()}"


class Visualization:
    def __init__(
        self,
        model_name: str,
        baseline_name: str,
        predictions: xr.Dataset,
        observations: xr.Dataset,
        baselines: xr.Dataset,
        model_metrics: xr.Dataset,
        baseline_metrics: xr.Dataset,
        visualization_types: dict,
        output_directory: Path,
    ):
        self.model_name, self.baseline_name = model_name, baseline_name
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
            "time_series_for_a_single_site": self.visualize_time_series_for_a_single_site,
            "error_time_series_for_a_single_site": self.visualize_error_time_series_for_a_single_site,
            "error_distribution_for_a_single_site": self.visualize_error_distribution_for_a_single_site,
            "boxplot_for_a_single_site": self.visualize_boxplot_for_a_single_site,
        }
        for (
            visualization_type_name,
            visualization_type_arg,
        ) in self.visualization_types.items():
            visualization_funct = visualization_dict_to_funct[visualization_type_name]
            visualization_funct(**visualization_type_arg)

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

    def visualize_metrics_global_maps(
        self, show_baseline: bool, color_palette=None, color_palette_signed=None
    ):
        """
        Visualize every single metric in the self.model_metrics.

        Parameters
        ----------
        show_baseline : bool
            Flag indicating whether to include baseline metrics in the visualization.
        color_palette : list, optional
            List of colors for the visualizations when only positive or negative values
            are shown.
            If None, default colors will be used.
        color_palette_signed : list, optional
            List of colors for the visualizations when the legends go from
            negative to positive.
            If None, default colors will be used.

        Returns
        -------
        None
        """
        if color_palette is None:
            color_palette = [
                "#d8f3dc",
                "#b7e4c7",
                "#95d5b2",
                "#74c69d",
                "#52b788",
                "#40916c",
                "#2d6a4f",
                "#1b4332",
                "#081c15",
            ]

        if color_palette_signed is None:
            color_palette_signed = [
                "#006d77",
                "#83c5be",
                "#edf6f9",
                "#ffddd2",
                "#e29578",
            ]

        if not show_baseline:
            for metric_var in list(self.model_metrics.data_vars):
                fig, ax = plt.subplots(
                    figsize=(15, 10),
                    ncols=1,
                    nrows=1,
                    subplot_kw={"projection": ccrs.PlateCarree()},
                )
                ax.coastlines()
                self.model_metrics[metric_var].plot(
                    ax=ax,
                    x="x",
                    y="y",
                    cmap=mcolors.LinearSegmentedColormap.from_list("", color_palette),
                )
                ax.set_title(
                    f"{self.model_name.upper()} - {metric_var.upper()} "
                    f"(Mean: {round(float(self.model_metrics[metric_var].mean()), 2)} "
                    f"ºC)"
                )
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
                if minvalue < 0 and maxvalue > 0:
                    cmap = mcolors.LinearSegmentedColormap.from_list(
                        "", color_palette_signed
                    )
                    if abs(minvalue) > abs(maxvalue):
                        maxvalue = abs(minvalue)
                    else:
                        minvalue = maxvalue * (-1)
                else:
                    cmap = mcolors.LinearSegmentedColormap.from_list("", color_palette)

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
                            round(x, 2)
                            for x in np.linspace(minvalue, maxvalue, len(color_palette))
                        ],
                    },
                    cmap=cmap,
                )
                ax1.coastlines()
                ax1.set_title(
                    f"{self.model_name.upper()} - {metric_var.upper()} "
                    f"(Mean: {round(float(self.model_metrics[metric_var].mean()), 2)})"
                )
                ax1.set_label(f"{metric_var.upper()}")
                self.baseline_metrics[metric_var].plot(
                    ax=ax2,
                    x="longitude",
                    y="latitude",
                    vmin=minvalue,
                    vmax=maxvalue,
                    cbar_kwargs={
                        "ticks": [
                            round(x, 2)
                            for x in np.linspace(minvalue, maxvalue, len(color_palette))
                        ],
                        "spacing": "proportional",
                    },
                    cmap=cmap,
                )
                ax2.coastlines()
                ax2.set_title(
                    f"{self.baseline_name.upper()} - {metric_var.upper()} "
                    f"(Mean: "
                    f"{round(float(self.baseline_metrics[metric_var].mean()), 2)})"
                )
                ax2.set_label(f"{metric_var.upper()}")
                plt.tight_layout()
                output_path = self.get_output_path("metric_global_map", metric_var)
                plt.savefig(output_path, bbox_inches="tight")
                plt.close()

    def visualize_sample_observation_vs_prediction(
        self, number_of_samples: int, color_palette=None
    ):
        """
        Visualize the prediction, baseline, and observation for a given number of samples.

        Parameters
        ----------
        number_of_samples : int
            Number of samples to visualize.
        color_palette : list, optional
            List of colors for the visualizations.
            If None, default colors will be used.

        Returns
        -------
        None
        """
        if color_palette is None:
            color_palette = [
                "#d8f3dc",
                "#b7e4c7",
                "#95d5b2",
                "#74c69d",
                "#52b788",
                "#40916c",
                "#2d6a4f",
                "#1b4332",
                "#081c15",
            ]

        samples_done = []
        for i in range(number_of_samples):
            random_sample = random.randint(0, len(self.predictions.time.values))
            while random_sample in samples_done:
                random_sample = random.randint(0, len(self.predictions.time.values))
            samples_done.append(random_sample)
            prediction = self.predictions.isel(time=random_sample)
            baseline = self.baselines.isel(time=random_sample)
            observation = self.observations.isel(time=random_sample)
            time_value = str(pd.to_datetime(observation.time.values))

            min_value = min(
                prediction.variable.min().values,
                baseline.variable.min().values,
                observation.variable.min().values,
            )
            max_value = max(
                prediction.variable.max().values,
                baseline.variable.max().values,
                observation.variable.max().values,
            )

            fig, axs = plt.subplots(
                ncols=1,
                nrows=3,
                figsize=(10, 15),
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
            ax1, ax2, ax3 = axs

            for ax, data, title in zip(
                [ax1, ax2, ax3],
                [prediction, baseline, observation],
                [
                    f"{self.model_name.upper()} - {time_value}",
                    f"{self.baseline_name.upper()}  - {time_value}",
                    f"{'Observation'.upper()} - {time_value}",
                ],
            ):
                ax.coastlines()
                cmap = mcolors.LinearSegmentedColormap.from_list("", color_palette)

                plot = data.variable.plot(
                    ax=ax,
                    x="longitude",
                    y="latitude",
                    vmin=min_value,
                    vmax=max_value,
                    cmap=cmap,
                    add_colorbar=False,
                )
                ax.set_title(title)

            # Add a single colorbar for all three plots
            cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
            cbar = plt.colorbar(plot, cax=cbar_ax)
            cbar.set_ticks(np.linspace(min_value, max_value, len(color_palette)))
            cbar.set_label("Temperature (ºC)")

            plt.tight_layout()  # Adjust the layout to accommodate the colorbar
            output_path = self.get_output_path("sample_observation_vs_prediction", i)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()

    def visualize_time_series_for_a_single_site(
        self,
        locations: list,
        temporal_subset=None,
        color_palette=None,
        aggregate_by=None,
    ):
        """
        Create time series plots for a single site with different aggregation periods.

        Parameters
        ----------
        locations : list
            List of locations to visualize.
        temporal_subset : slice or tuple of slice, optional
            Temporal subset to plot. If None, plot the entire time series.
            Example: To plot data from '2023-01-01' to '2023-03-31',
                     use slice('2023-01-01', '2023-04-01').
        color_palette : list, optional
            List of colors for the visualizations. If None, default colors will be used.
        aggregate_by : list, optional
            List of time periods to aggregate the data.
            Examples: ["1D", "7D", "15D", "1M"].

        Returns
        -------
        None
        """
        if color_palette is None:
            color_palette = ["#001219", "#ee9b00", "#9b2226"]

        data_merged = xr.merge(
            [
                self.observations.rename_vars({"variable": "obs"}),
                self.predictions.rename_vars({"variable": "preds"}),
                self.baselines.rename_vars({"variable": "baseline"}),
            ]
        )

        for location in locations:
            location_obj = Location(**location)
            location_data = data_merged.sel(
                longitude=location_obj.lon, latitude=location_obj.lat, method="nearest"
            )

            if temporal_subset:
                location_data = location_data.sel(time=temporal_subset)

            # Create separate plots for each aggregation period
            for agg_period in aggregate_by:
                if agg_period == "1D":
                    agg_label = "Daily"
                    location_data_agg = location_data.resample(time="1D").mean()
                elif agg_period == "7D":
                    agg_label = "Weekly"
                    location_data_agg = location_data.resample(time="1W").mean()
                elif agg_period == "15D":
                    agg_label = "Biweekly"
                    location_data_agg = location_data.resample(time="15D").mean()
                elif agg_period == "1M":
                    agg_label = "Monthly"
                    location_data_agg = location_data.resample(time="1M").mean()
                else:
                    raise ValueError("Invalid value for aggregate_by")

                max_of_maxs = location_data_agg.max().to_array().values.max()
                min_of_mins = location_data_agg.min().to_array().values.min()

                # Create a larger figure
                plt.figure(figsize=(15, 10))  # Adjust the size as needed

                # Points for the scatter plot
                plt.plot(
                    location_data_agg.time.values,
                    location_data_agg["obs"].values,
                    label="Observation".upper(),
                    color=color_palette[0],
                )
                plt.plot(
                    location_data_agg.time.values,
                    location_data_agg["preds"].values,
                    label=self.model_name.upper(),
                    color=color_palette[1],
                    alpha=0.7,
                )
                plt.plot(
                    location_data_agg.time.values,
                    location_data_agg["baseline"].values,
                    label=self.baseline_name.upper(),
                    color=color_palette[2],
                    alpha=0.7,
                )

                plt.ylim([min_of_mins, max_of_maxs])
                plt.title(
                    f"{agg_label} time series at "
                    f"{location['name'].replace('_', ' ').replace('-', ' - ')}."
                )
                plt.xlabel("Time", fontsize=14)  # Add an appropriate x-axis label
                plt.ylabel(
                    "Temperature (ºC)", fontsize=14
                )  # Add an appropriate y-axis label
                plt.xticks(
                    rotation=45, fontsize=12
                )  # Rotate x-axis labels to avoid overlapping
                plt.xticks(fontsize=12)
                plt.legend()

                output_path = self.get_output_path(
                    f"{agg_label.lower()}_time_series_for",
                    location_obj,
                )
                plt.savefig(output_path, bbox_inches="tight")
                plt.close()

    def visualize_error_time_series_for_a_single_site(
        self,
        locations: list,
        temporal_subset=None,
        color_palette=None,
        aggregate_by=None,
    ):
        """
        Create error time series plots for a single site.

        Parameters
        ----------
        locations : list
            List of locations to visualize.
        temporal_subset : slice or tuple of slice, optional
            Temporal subset to plot. If None, plot the entire time series.
            Example: To plot data from '2023-01-01' to '2023-03-31',
                     use slice('2023-01-01', '2023-04-01').
        color_palette : list, optional
            List of colors for the visualizations. If None, default colors will be used.
        aggregate_by : list, optional
            List of time periods to aggregate the data.
            Examples: ["1D", "7D", "15D", "1M"].

        Returns
        -------
        None
        """
        if color_palette is None:
            color_palette = ["#ee9b00", "#9b2226"]

        data_merged = xr.merge(
            [
                self.observations.rename_vars({"variable": "obs"}),
                self.predictions.rename_vars({"variable": "preds"}),
                self.baselines.rename_vars({"variable": "baseline"}),
            ]
        )

        for location in locations:
            location_obj = Location(**location)
            location_data = data_merged.sel(
                longitude=location_obj.lon, latitude=location_obj.lat, method="nearest"
            )

            if temporal_subset:
                location_data = location_data.sel(time=temporal_subset)

            for agg_period in aggregate_by:
                if agg_period == "1D":
                    agg_label = "Daily"
                    location_data_agg = location_data.resample(time="1D").mean()
                elif agg_period == "7D":
                    agg_label = "Weekly"
                    location_data_agg = location_data.resample(time="1W").mean()
                elif agg_period == "15D":
                    agg_label = "Biweekly"
                    location_data_agg = location_data.resample(time="15D").mean()
                elif agg_period == "1M":
                    agg_label = "Monthly"
                    location_data_agg = location_data.resample(time="1M").mean()
                else:
                    raise ValueError("Invalid value for aggregate_by")

                error_preds = location_data_agg["preds"] - location_data_agg["obs"]
                error_baseline = (
                    location_data_agg["baseline"] - location_data_agg["obs"]
                )

                plt.figure(figsize=(15, 10))
                plt.plot(
                    location_data_agg.time.values,
                    error_preds.values,
                    label=f"{self.model_name.upper()}",
                    color=color_palette[0],
                )
                plt.plot(
                    location_data_agg.time.values,
                    error_baseline.values,
                    label=f"{self.baseline_name.upper()}",
                    color=color_palette[1],
                )

                plt.axhline(y=0, color="black", linestyle="dashed")

                plt.ylim(
                    [
                        min(np.min(error_preds), np.min(error_baseline)),
                        max(np.max(error_preds), np.max(error_baseline)),
                    ]
                )

                plt.title(
                    f"{agg_label} error time series at "
                    f"{location['name'].replace('_', ' ').replace('-', ' - ')}."
                )
                plt.xlabel("Time", fontsize=14)
                plt.ylabel("Error (ºC)", fontsize=14)
                plt.xticks(rotation=45, fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend()

                output_path = self.get_output_path(
                    f"{agg_label.lower()}_error_time_series_for",
                    location_obj,
                )
                plt.savefig(output_path, bbox_inches="tight")
                plt.close()

    def visualize_error_distribution_for_a_single_site(
        self,
        locations: list,
        temporal_subset=None,
        color_palette=None,
    ):
        """
        Create error distribution plots for a single site.

        Parameters
        ----------
        locations : list
            List of locations to visualize.
        temporal_subset : slice or tuple of slice, optional
            Temporal subset to plot. If None, plot the entire time series.
            Example: To plot data from '2023-01-01' to '2023-03-31',
                     use slice('2023-01-01', '2023-04-01').
        color_palette : list, optional
            List of colors for the visualizations. If None, default colors will be used.

        Returns
        -------
        None
        """
        if color_palette is None:
            color_palette = ["#ee9b00", "#9b2226"]

        data_merged = xr.merge(
            [
                self.observations.rename_vars({"variable": "obs"}),
                self.predictions.rename_vars({"variable": "preds"}),
                self.baselines.rename_vars({"variable": "baseline"}),
            ]
        )

        for location in locations:
            location_obj = Location(**location)
            location_data = data_merged.sel(
                longitude=location_obj.lon, latitude=location_obj.lat, method="nearest"
            )

            if temporal_subset:
                location_data = location_data.sel(time=temporal_subset)

            error_preds = location_data["preds"] - location_data["obs"]
            error_baseline = location_data["baseline"] - location_data["obs"]

            # Create a larger figure
            plt.figure(figsize=(15, 10))

            # Plot the PDF of error values for predictions
            plt.hist(
                error_preds.values,
                bins=30,
                density=True,
                alpha=0.5,
                color=color_palette[0],
                label=f"{self.model_name.upper()}",
            )

            # Plot the PDF of error values for baselines
            plt.hist(
                error_baseline.values,
                bins=30,
                density=True,
                alpha=0.5,
                color=color_palette[1],
                label=f"{self.baseline_name.upper()}",
            )

            # Add a normal distribution fit to the predictions error
            mu_preds, std_preds = norm.fit(error_preds)
            x = np.linspace(min(error_preds), max(error_preds), 100)
            p = norm.pdf(x, mu_preds, std_preds)
            plt.plot(x, p, color=color_palette[0])

            # Add a normal distribution fit to the baselines error
            mu_baseline, std_baseline = norm.fit(error_baseline)
            x = np.linspace(min(error_baseline), max(error_baseline), 100)
            p = norm.pdf(x, mu_baseline, std_baseline)
            plt.plot(x, p, color=color_palette[1])

            plt.axvline(x=0, color="black", linestyle="dashed")

            # Add vertical lines at the mean value of the error distribution
            plt.axvline(
                x=mu_preds, color=color_palette[0], linestyle="solid", alpha=0.6
            )
            plt.axvline(
                x=mu_baseline, color=color_palette[1], linestyle="solid", alpha=0.6
            )

            plt.title(
                f"Error Distribution at "
                f"{location['name'].replace('_', ' ').replace('-', ' - ')}."
            )
            plt.xlabel("Error (ºC)", fontsize=14)  # Increase font size
            plt.ylabel("Probability Density", fontsize=14)  # Increase font size
            plt.xticks(fontsize=12)  # Increase x-axis tick font size
            plt.yticks(fontsize=12)  # Increase y-axis tick font size

            plt.legend()

            output_path = self.get_output_path(
                "error_distribution_for",
                location_obj,
            )
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()

    def visualize_boxplot_for_a_single_site(
        self,
        locations: list,
        group_by=None,
        color_palette=None,
    ):
        """
        Create a box plot for a single site.

        Parameters
        ----------
        locations : list
            List of locations to visualize.
        group_by : str, optional
            The grouping option for creating boxplots.
            Options: "hour", "month", or "season".
            Default is "season".
        color_palette : list, optional
            List of three colors for the "Data Type" hue in the box plot.
            Default: ['#4c956c', '#fefee3', '#ffc9b9']

        Returns
        -------
        None
        """
        if group_by is None:
            group_by = ["hour", "month", "season"]

        original_variable = "Temperature (ºC)"
        data_merged = xr.merge(
            [
                self.observations.rename_vars({"variable": "OBSERVATIONS"}),
                self.predictions.rename_vars({"variable": self.model_name.upper()}),
                self.baselines.rename_vars({"variable": self.baseline_name.upper()}),
            ]
        )

        if color_palette is None:
            color_palette = ["#4c956c", "#fefee3", "#ffc9b9"]

        # Loop through the group_by options
        for group_by_option in group_by:
            if group_by_option not in ["hour", "month", "season"]:
                raise ValueError(
                    "Invalid group_by option. Choose from 'hour', 'month', or 'season'."
                )

            for location in locations:
                location_obj = Location(**location)
                location_data = data_merged.sel(
                    longitude=location_obj.lon,
                    latitude=location_obj.lat,
                    method="nearest",
                )
                location_data_df = (
                    location_data.to_dataframe()
                    .drop(columns=["longitude", "latitude"])
                    .reset_index()
                )

                if group_by_option == "hour":
                    # Group data by hour of the day
                    group_by_option_str = "Hourly"
                    location_data_df["hour"] = pd.to_datetime(
                        location_data_df.time.values
                    ).hour
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
                elif group_by_option == "month":
                    # Group data by month
                    group_by_option_str = "Monthly"
                    location_data_df["month"] = pd.to_datetime(
                        location_data_df.time.values
                    ).month
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
                    group_by_option_str = "Seasonal"
                    location_data_df["season"] = [
                        (month % 12 // 3 + 1)
                        for month in list(
                            pd.to_datetime(location_data_df.time.values).month
                        )
                    ]
                    seasoninttostr = {
                        1: "Winter",
                        2: "Spring",
                        3: "Summer",
                        4: "Autumn",
                    }
                    location_data_df["Season"] = [
                        seasoninttostr[x] for x in location_data_df.season.values
                    ]
                    location_data_df = location_data_df.drop(columns=["time", "season"])

                location_data_df = (
                    location_data_df.set_index(group_by_option.capitalize())
                    .stack()
                    .reset_index()
                    .rename(columns={"level_1": "Data Type", 0: original_variable})
                )

                fig, ax = plt.subplots(figsize=(15, 10), ncols=1, nrows=1)

                sns.boxplot(
                    ax=ax,
                    data=location_data_df,
                    x=group_by_option.capitalize(),
                    y=original_variable,
                    hue="Data Type",
                    palette=color_palette,
                )

                plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

                time_cov_str = (
                    f"from "
                    f"{pd.to_datetime(data_merged.time.values[0]).strftime('%d-%m-%Y')} "
                    f"to "
                    f"{pd.to_datetime(data_merged.time.values[-1]).strftime('%d-%m-%Y')}"
                )
                plt.title(
                    f"{group_by_option_str} box plot at "
                    f"{location['name'].replace('_', ' ').replace('-', ' - ')}, "
                    f"{time_cov_str}."
                )
                output_path = self.get_output_path(
                    f"{group_by_option}_boxplot_for",
                    location_obj,
                )
                plt.savefig(output_path, bbox_inches="tight")
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
