from typing import Callable, Type

import evaluate
import torch
from tqdm import tqdm

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
