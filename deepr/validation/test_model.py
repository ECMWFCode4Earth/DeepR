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


def compute_errors_vs_baseline(
    model: Type[torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    baseline: str = "bicubic",
    scaler_func: Callable = None,
):
    count = 0
    abs_errors = torch.zeros(dataloader.dataset.output_shape)
    sq_errors = torch.zeros(dataloader.dataset.output_shape)
    abs_errors_bi = torch.zeros(dataloader.dataset.output_shape)
    sq_errors_bi = torch.zeros(dataloader.dataset.output_shape)
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

    return mae, mse, mae_bi, mse_bi


def show_samples(
    model,
    dataloader: torch.utils.data.DataLoader,
    local_dir: str,
    scaler_func: Callable = None,
    baseline: str = "bicubic",
):
    era5, cerra, times = next(iter(dataloader))
    with torch.no_grad():
        pred_nn = model(era5, return_dict=False)[0]
    samples_base = torch.nn.functional.interpolate(
        era5[..., 6:-6, 6:-6], scale_factor=5, mode=baseline
    )

    if scaler_func is not None:
        cerra = scaler_func(cerra, times[:, 2])
        samples_base = scaler_func(samples_base, times[:, 2])
        pred_nn = scaler_func(pred_nn, times[:, 2])

    plot_2_model_comparison(
        cerra[0, 0],
        samples_base[0, 0],
        pred_nn[0, 0],
        matrix_names=["CERRA", baseline.capitalize(), model.__class__.__name__],
        metric_name="ºC",
        date=f"{times[0, 0]:d}H {times[0, 1]:d}-{times[0, 2]:d}-{times[0, 3]:d}",
        filename=Path(local_dir) / "pred_comparison.png",
    )


def test_model(
    model,
    dataset: torch.utils.data.IterableDataset,
    batch_size: int = os.getenv("BATCH_SIZE", 4),
    hf_repo_name: str = None,
    label_scaler: XarrayStandardScaler = None,
    baseline: str = "nearest",
):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, pin_memory=True)
    scaler_func = None if label_scaler is None else label_scaler.apply_inverse_scaler

    local_dir = f"hf-{model.__class__.__name__}-evaluation"
    if hf_repo_name is not None:
        repo = Repository(
            local_dir, clone_from=hf_repo_name, token=os.getenv("HF_TOKEN")
        )
        repo.git_pull()

    # Show samples compared with other models
    show_samples(model, dataloader, local_dir, scaler_func, baseline)

    # Obtain error maps
    mae, mse, mae_base, mse_base = compute_errors_vs_baseline(
        model, dataloader, baseline, scaler_func
    )
    names = [model.__class__.__name__, baseline]
    plot_2_maps_comparison(
        mse, mse_base, names, "MSE (ºC)", f"{local_dir}/mse_vs_{baseline}.png", vmin=0
    )
    plot_2_maps_comparison(
        mae, mae_base, names, "MAE (ºC)", f"{local_dir}/mae_vs_{baseline}.png", vmin=0
    )

    test_metrics = compute_and_upload_metrics(
        model, dataloader, hf_repo_name, scaler_func
    )
    evaluate.save(tmpdir, experiment=experiment_name, **test_metrics)

    if hf_repo_name is not None:
        repo.push_to_hub(
            repo_id=hf_repo_name,
            commit_message=f"Tests on {dataset.init_date}-{dataset.end_date}",
            blocking=True,
        )
