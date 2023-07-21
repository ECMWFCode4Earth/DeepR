from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def calculate_metric_mean(tensors: Dict, land_mask: Tuple[bool, np.ndarray]) -> list:
    """
    Calculate the mean value of a metric for each tensor in the given dictionary.

    Parameters
    ----------
        tensors (dict): A dictionary where each key corresponds to a tensor.
                        The values of the dictionary are tensors (2D arrays).
        land_mask (numpy.array): A 2D numpy array representing the land mask.

    Returns
    -------
        list: A list containing the mean values of the metric for each tensor.
    """
    # Convert the dictionary values to a list of tensors
    tensor_list = list(tensors.values())

    # Filter out the values where the land mask is 0
    if land_mask is not None:
        if land_mask[0] == "land":
            tensor_list = [
                np.where(land_mask[1] == 1, tensor.numpy(), np.nan)
                for tensor in tensor_list
            ]
        elif land_mask[0] == "sea":
            tensor_list = [
                np.where(land_mask[1] == 0, tensor.numpy(), np.nan)
                for tensor in tensor_list
            ]
        else:
            raise NotImplementedError

    # Calculate the mean value of the metric for each tensor and store it in a list
    mean_values = [np.nanmean(np.abs(tensor)) for tensor in tensor_list]

    return mean_values


def plot_rose(
    metric: dict,
    metric_baseline: dict,
    land_mask: Tuple[bool, np.array],
    names: list,
    custom_colors: list,
    title: str,
    output_path: str,
) -> None:
    """
    Create two rose plots side by side for the given metric dictionaries.

    Parameters
    ----------
        metric (dict): A dictionary where each key corresponds to a tensor.
                       The values of the dictionary are tensors (2D arrays).
        metric_baseline (dict): Another dictionary with the same structure as 'metric'
                                representing the baseline metric values.
        land_mask (numpy.array): A 2D numpy array representing the land mask.
        title (str): The title of the rose plots.
        output_path (str): The file path where the plot will be saved.
    """
    # Get the keys from the dictionaries and sort them in ascending order
    keys_metric = sorted(list(metric.keys()), reverse=True)
    if metric_baseline is not None:
        keys_baseline = sorted(list(metric_baseline.keys()), reverse=True)

    # Calculate the angle step for each key (clockwise direction)
    angle_step = 360.0 / len(keys_metric)

    # Assign unique angles to each key, starting at 90 degrees and moving clockwise
    angles_metric = ((np.arange(len(keys_metric)) * angle_step) + 90 + angle_step) % 360
    if metric_baseline is not None:
        angles_baseline = (
            (np.arange(len(keys_baseline)) * angle_step) + 90 + angle_step
        ) % 360

    # Calculate the mean metric values for the 'metric' dictionary
    mean_values_metric = calculate_metric_mean(metric, land_mask)

    # Calculate the mean metric values for the 'metric_baseline' dictionary
    if metric_baseline is not None:
        mean_values_baseline = calculate_metric_mean(metric_baseline, land_mask)

    # Set the width of the bars (adjust the value as needed)
    bar_width = 10.0

    # Set the larger tick sizes
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16

    # Create the rose plots side by side
    plt.figure(figsize=(18, 12))

    ax = plt.subplot(polar=True)

    # Calculate the positions for the bars
    positions_metric = np.radians(angles_metric) - np.radians(bar_width / 2)
    if metric_baseline is not None:
        positions_baseline = np.radians(angles_baseline) + np.radians(bar_width / 2)

    if metric_baseline is not None:
        ax.bar(
            positions_baseline,
            mean_values_baseline,
            width=np.radians(bar_width),
            align="center",
            alpha=0.8,
            label=names[1].capitalize(),
            color=custom_colors[1],
        )

    ax.bar(
        positions_metric,
        mean_values_metric,
        width=np.radians(bar_width),
        align="center",
        alpha=0.8,
        label=names[0].capitalize(),
        color=custom_colors[0],
    )

    # Remove the last circumference (circular axis line)
    ax.spines["polar"].set_visible(False)

    plt.title(title, fontsize=20)
    plt.thetagrids(angles_baseline, labels=keys_baseline)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
