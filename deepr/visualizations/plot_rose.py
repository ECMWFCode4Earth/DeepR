import matplotlib.pyplot as plt
import numpy as np


def calculate_metric_mean(tensors: dict) -> list:
    """
    Calculate the mean value of a metric for each tensor in the given dictionary.

    Parameters
    ----------
        tensors (dict): A dictionary where each key corresponds to a tensor.
                        The values of the dictionary are tensors (2D arrays).

    Returns
    -------
        list: A list containing the mean values of the metric for each tensor.
    """
    # Convert the dictionary values to a list of tensors
    tensor_list = list(tensors.values())

    # Calculate the mean value of the metric for each tensor and store it in a list
    mean_values = [np.mean(np.abs(tensor.numpy())) for tensor in tensor_list]

    return mean_values


def plot_rose(
    metric: dict, metric_baseline: dict, title: str, output_path: str
) -> None:
    """
    Create two rose plots side by side for the given metric dictionaries.

    Parameters
    ----------
        metric (dict): A dictionary where each key corresponds to a tensor.
                       The values of the dictionary are tensors (2D arrays).
        metric_baseline (dict): Another dictionary with the same structure as 'metric'
                                representing the baseline metric values.
        title (str): The title of the rose plots.
        output_path (str): The file path where the plot will be saved.
    """
    # Get the keys from the dictionaries
    keys_metric = list(metric.keys())
    keys_baseline = list(metric_baseline.keys())

    # Assign unique angles to each key
    angles_metric = np.arange(len(keys_metric)) * (360.0 / len(keys_metric))
    angles_baseline = np.arange(len(keys_baseline)) * (360.0 / len(keys_baseline))

    # Calculate the mean metric values for the 'metric' dictionary
    mean_values_metric = calculate_metric_mean(metric)

    # Calculate the mean metric values for the 'metric_baseline' dictionary
    mean_values_baseline = calculate_metric_mean(metric_baseline)

    # Set the width of the bars (adjust the value as needed)
    bar_width = 10.0

    # Create the rose plots side by side
    plt.figure(figsize=(18, 8))

    plt.subplot(1, 2, 1, polar=True)
    plt.bar(
        np.radians(angles_metric),
        mean_values_metric,
        width=np.radians(bar_width),
        align="center",
        alpha=0.7,
    )
    plt.thetagrids(angles_metric, labels=keys_metric)
    plt.title(title + " - Metric", fontsize=14)

    plt.subplot(1, 2, 2, polar=True)
    plt.bar(
        np.radians(angles_baseline),
        mean_values_baseline,
        width=np.radians(bar_width),
        align="center",
        alpha=0.7,
    )
    plt.thetagrids(angles_baseline, labels=keys_baseline)
    plt.title(title + " - Metric Baseline", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path)
