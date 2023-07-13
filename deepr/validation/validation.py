import os
import tempfile
from pathlib import Path
from typing import Callable, Type

import evaluate
import torch
from huggingface_hub import Repository
from tqdm import tqdm

from deepr.data.scaler import XarrayStandardScaler
from deepr.visualizations.plot_maps import (
    plot_2_maps_comparison,
    plot_2_model_comparison,
)

tmpdir = tempfile.mkdtemp(prefix="test-")

experiment_name = "Test Neural Network"
metric_to_repo = {
    "MSE": "mse",
    "R2": "r_squared",
    "SMAPE": "smape",
    "PSNR": "jpxkqx/peak_signal_to_noise_ratio",
    "SSIM": "jpxkqx/structural_similarity_index_measure",
    "SRE": "jpxkqx/signal_to_reconstruction_error",
}


def compute_and_upload_metrics(
    model: Type[torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    hf_repo_name: str = None,
    scaler_func: Callable = None,
):
    """Compute and upload a set of metrics.

    The metrics computed in this function are:
    - MSE: Mean Squared Error of the predictions.
    - R2: Pearson Correlation (R²) coefficient of the predictions.
    - SMAPE: Symmetric Mean Absolute Percentage Error of the predictions.
    - PSNR: Peak Signal to Noise Ratio of the predictions.
    - SSIM: Structural Similarity Index Measure of the predictions.
    - SRE: Signal to Reconstruction Error of the predictions.
    """
    # Load metrics over all dataset
    mse = evaluate.load("mse", "multilist")
    # r2 = evaluate.load("pearsonr", "multilist")
    smape = evaluate.load("smape", "multilist")
    psnr = evaluate.load("jpxkqx/peak_signal_to_noise_ratio", "multilist")
    ssim = evaluate.load("jpxkqx/structural_similarity_index_measure", "multilist")
    sre = evaluate.load("jpxkqx/signal_to_reconstruction_error", "multilist")

    progress_bar = tqdm(total=len(dataloader), desc="Batch ")
    max_pred, min_pred, max_true, min_true = -999, 999, -999, 999
    for era5, cerra, times in dataloader:
        # Predict the noise residual
        with torch.no_grad():
            pred = model(era5, return_dict=False)[0]
            if scaler_func is not None:
                pred = scaler_func(pred, times[:, 2])
                cerra = scaler_func(cerra, times[:, 2])

            mse.add_batch(
                references=cerra.reshape((cerra.shape[0], -1)),
                predictions=pred.reshape((pred.shape[0], -1)),
            )
            # r2.add_batch(
            #    references=cerra.reshape((cerra.shape[0], -1)),
            #    predictions=pred.reshape((pred.shape[0], -1)),
            # )
            smape.add_batch(
                references=cerra.reshape((cerra.shape[0], -1)),
                predictions=pred.reshape((pred.shape[0], -1)),
            )
            psnr.add_batch(references=cerra, predictions=pred)
            ssim.add_batch(references=cerra, predictions=pred)
            sre.add_batch(references=cerra, predictions=pred)
            max_pred, min_pred = max(max_pred, pred.max()), min(min_pred, pred.min())
            max_true, min_true = max(max_true, cerra.max()), min(min_true, cerra.min())
        progress_bar.update(1)
    progress_bar.close()

    # Compute Metrics
    data_range = float(max(max_pred, max_true) - min(min_pred, min_true))
    test_metrics = {
        "MSE": mse.compute()["mse"],
        # "R2": r2.compute()["pearsonr"],
        "SMAPE": smape.compute()["smape"],
        "PSNR": psnr.compute(data_range=data_range),
        "SSIM": ssim.compute(data_range=data_range, channel_axis=0),  # ignore batch dim
        "SRE": sre.compute()["Signal-to-Reconstruction Error"],
    }
    for name, metric_val in test_metrics.items():
        print(f"Test {name}: {metric_val:.2f}")

    if hf_repo_name is not None:
        for name, metric_val in test_metrics.items():
            evaluate.push_to_hub(
                model_id=hf_repo_name,
                metric_type=metric_to_repo[name],
                metric_name=name,
                metric_value=metric_val,
                dataset_type="era5",
                dataset_name="ERA5+CERRA",
                dataset_split="test",
                task_type="image-to-image",
                task_name="Super Resolution",
            )

    return test_metrics


def compute_model_and_baseline_errors(
    model: Type[torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    baseline: str = "bicubic",
    scaler_func: Callable = None,
):
    """
    Compute the model and baseline errors.

    It makes it by comparing the predictions with the ground truth labels.

    Parameters
    ----------
    model : Type[torch.nn.Module]
        The neural network model.
    dataloader : torch.utils.data.DataLoader
        The data loader used to fetch the data.
    baseline : str, optional
        The mode used for baseline interpolation, by default "bicubic".
    scaler_func : Callable, optional
        A scaling function to apply on the data, by default None.

    Returns
    -------
    mae : torch.Tensor
        Mean Absolute Error (MAE) between the model predictions and the
        ground truth labels.
    mse : torch.Tensor
        Mean Squared Error (MSE) between the model predictions and the
        ground truth labels.
    mae_bi : torch.Tensor
        MAE between the baseline predictions and the ground truth labels.
    mse_bi : torch.Tensor
        MSE between the baseline predictions and the ground truth labels.
    improvement : torch.Tensor
        Percentage of improvement of the error from the model to the baseline.

    """
    count = 0
    keys = [0, 3, 6, 9, 12, 15, 18, 21, "all"]
    abs_errors, sq_errors, abs_errors_bi, sq_errors_bi, improvement = {}, {}, {}, {}, {}
    for key in keys:
        abs_errors[key] = torch.zeros(dataloader.dataset.output_shape)
        sq_errors[key] = torch.zeros(dataloader.dataset.output_shape)
        abs_errors_bi[key] = torch.zeros(dataloader.dataset.output_shape)
        sq_errors_bi[key] = torch.zeros(dataloader.dataset.output_shape)
        improvement[key] = torch.zeros(dataloader.dataset.output_shape)
    progress_bar = tqdm(total=len(dataloader), desc="Batch ")

    for era5, cerra, times in dataloader:
        # Predict the noise residual
        with torch.no_grad():
            pred = model(era5, return_dict=False)[0]

        pred_bi = torch.nn.functional.interpolate(
            era5[..., 6:-6, 6:-6], scale_factor=5, mode=baseline
        )

        if scaler_func is not None:
            pred = scaler_func(pred, times[:, 2])
            pred_bi = scaler_func(pred_bi, times[:, 2])
            cerra = scaler_func(cerra, times[:, 2])

        count += era5.shape[0]
        error = pred - cerra
        error_bi = pred_bi - cerra
        abs_errors += torch.sum(torch.abs(error), (0, 1))
        sq_errors += torch.sum(error**2, (0, 1))
        abs_errors_bi += torch.sum(torch.abs(error_bi), (0, 1))
        sq_errors_bi += torch.sum(error_bi**2, (0, 1))
        progress_bar.update(1)

    progress_bar.close()

    mae = abs_errors / count
    mse = sq_errors / count
    mae_bi = abs_errors_bi / count
    mse_bi = sq_errors_bi / count

    return mae, mse, mae_bi, mse_bi, improvement


def sample_observation_versus_prediction(
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
        date_str = f"{times[0, 0]:d}H {times[0, 1]:d}-{times[0, 2]:d}-{times[0, 3]:d}"
        plot_2_model_comparison(
            cerra[0, 0],
            samples_base[0, 0],
            pred_nn[0, 0],
            matrix_names=["CERRA", baseline.capitalize(), model.__class__.__name__],
            metric_name="ºC",
            date=date_str,
            filename=filename,
        )
        samples_get += 1
        if samples_get == number_of_samples:
            break


def validate_model(
    model,
    dataset: torch.utils.data.IterableDataset,
    config: dict,
    batch_size: int = os.getenv("BATCH_SIZE", 4),
    hf_repo_name: str = None,
    label_scaler: XarrayStandardScaler = None,
):
    """
    Validate the model.

    It makes it by generating evaluation plots, computing error metrics, and uploading
    metrics to the Hugging Face Model Hub.

    Parameters
    ----------
    model : object
        The neural network model to validate.
    dataset : torch.utils.data.IterableDataset
        The dataset used for validation.
    config : dict
        Configuration settings for the validation.
    batch_size : int, optional
        Batch size for data loading, by default 4.
    hf_repo_name : str, optional
        Hugging Face repository name, by default None.
    label_scaler : XarrayStandardScaler, optional
        Label scaler object for applying inverse scaling, by default None.

    Returns
    -------
    None

    """
    # Create data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, pin_memory=True)

    # Define scaler function if label scaler is provided
    scaler_func = None if label_scaler is None else label_scaler.apply_inverse_scaler

    # Define local directory for saving evaluation results
    local_dir = f"{config['output_directory']}/hf-{model.__class__.__name__}-evaluation"

    # Clone Hugging Face repository if provided
    if hf_repo_name is not None:
        repo = Repository(
            local_dir, clone_from=hf_repo_name, token=os.getenv("HF_TOKEN")
        )
        repo.git_pull()

    # Show samples compared with other models
    if config["visualizations"]["sample_observation_versus_prediction"] > 0:
        visualization_local_dir = f"{local_dir}/sample_observation_versus_prediction"
        os.makedirs(visualization_local_dir, exist_ok=True)
        sample_observation_versus_prediction(
            model, dataloader, visualization_local_dir, scaler_func, config["baseline"]
        )

    # Obtain error maps
    mae, mse, mae_base, mse_base, improvement = compute_model_and_baseline_errors(
        model, dataloader, config["baseline"], scaler_func
    )
    names = [model.__class__.__name__, config["baseline"]]
    plot_2_maps_comparison(
        mse,
        mse_base,
        names,
        "MSE (ºC)",
        f"{local_dir}/mse_vs_{config['baseline']}.png",
        vmin=0,
    )
    plot_2_maps_comparison(
        mae,
        mae_base,
        names,
        "MAE (ºC)",
        f"{local_dir}/mae_vs_{config['baseline']}.png",
        vmin=0,
    )

    # Compute and upload metrics to Hugging Face Model Hub
    test_metrics = compute_and_upload_metrics(
        model, dataloader, hf_repo_name, scaler_func
    )
    evaluate.save(tmpdir, experiment=experiment_name, **test_metrics)

    # Push changes to Hugging Face repository if provided
    if hf_repo_name is not None:
        repo.push_to_hub(
            repo_id=hf_repo_name,
            commit_message=f"Tests on {dataset.init_date}-{dataset.end_date}",
            blocking=True,
        )
