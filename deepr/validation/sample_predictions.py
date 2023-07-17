import os
import tempfile
from pathlib import Path
from typing import Callable, Type

import evaluate
import torch
from huggingface_hub import Repository
from tqdm import tqdm

from deepr.data.scaler import XarrayStandardScaler
from deepr.validation.nn_performance_metrics import (
    compute_and_upload_metrics, compute_model_and_baseline_errors
)
from deepr.visualizations.plot_maps import (
    plot_2_maps_comparison,
    plot_2_model_comparison,
)

def sample_observation_vs_prediction(
    model,
    dataloader: torch.utils.data.DataLoader,
    local_dir: str,
    scaler_func: Callable = None,
    baseline: str = "bicubic",
    number_of_samples: int = 10,
):
    """
    Generate and save a comparison plot of model predictions and baseline samples.

    Parameters
    ----------
    model : object
        The neural network model used for predictions.
    dataloader : torch.utils.data.DataLoader
        The data loader used to fetch the data.
    local_dir : str
        The directory where the plot will be saved.
    scaler_func : Callable, optional
        A scaling function to apply on the data, by default None.
    baseline : str, optional
        The mode used for baseline interpolation, by default "bicubic".
    number_of_samples : int, optional
        The number of samples to randomly select and compare, by default 10.

    Returns
    -------
    None

    """
    samples_get = 0
    for era5, cerra, times in dataloader:
        with torch.no_grad():
            pred_nn = model(era5, return_dict=False)[0]
        samples_base = torch.nn.functional.interpolate(
            era5[..., 6:-6, 6:-6], scale_factor=5, mode=baseline
        )

        if scaler_func is not None:
            cerra = scaler_func(cerra, times[:, 2])
            samples_base = scaler_func(samples_base, times[:, 2])
            pred_nn = scaler_func(pred_nn, times[:, 2])

        filename = Path(local_dir) / f"pred_comparison_{samples_get}.png"
        for i in range(len(times)):
            t_str = f"{times[i, 0]:d}H {times[i, 1]:d}-{times[i, 2]:d}-{times[i, 3]:d}"
            plot_2_model_comparison(
                cerra[i, 0],
                samples_base[i, 0],
                pred_nn[i, 0],
                matrix_names=["CERRA", baseline.capitalize(), model.__class__.__name__],
                metric_name="ºC",
                date=t_str,
                filename=filename,
            )
            samples_get += 1
            if samples_get == number_of_samples:
                return None
