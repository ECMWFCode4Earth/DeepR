import tempfile

import evaluate
import torch
from tqdm import tqdm

tmpdir = tempfile.mkdtemp(prefix="test-")

experiment_name = "Test Neural Network"
metric_to_repo = {
    "MSE": "mse",
    "PSNR": "jpxkqx/peak_signal_to_noise_ratio",
    "SSIM": "jpxkqx/structural_similarity_index_measure",
    "SRE": "jpxkqx/signal_to_reconstruction_error",
}


def test_model(
    model,
    dataset: torch.utils.data.IterableDataset,
    hparams: dict = None,
    batch_size: int = 4,
    hf_repo_name: str = None,
):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, pin_memory=True)

    # Load metrics
    mse = evaluate.load("mse", "multilist")
    psnr = evaluate.load("jpxkqx/peak_signal_to_noise_ratio", "multilist")
    ssim = evaluate.load("jpxkqx/structural_similarity_index_measure", "multilist")
    sre = evaluate.load("jpxkqx/signal_to_reconstruction_error", "multilist")

    progress_bar = tqdm(total=len(dataloader), desc="Batch ")
    max_pred, min_pred, max_true, min_true = -999, 999, -999, 999
    for era5, cerra, *times in dataloader:
        # Predict the noise residual
        with torch.no_grad():
            pred = model(era5, return_dict=False)[0]
            mse.add_batch(
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
        "PSNR": psnr.compute(data_range=data_range),
        "SSIM": ssim.compute(data_range=data_range, channel_axis=0),  # ignore batch dim
        "SRE": sre.compute()["Signal-to-Reconstruction Error"],
    }
    for name, metric_val in test_metrics.items():
        print(f"Test {name}: {metric_val:.2f}")

    if hparams is not None:
        evaluate.save(tmpdir, experiment=experiment_name, **test_metrics, **hparams)
    else:
        evaluate.save(tmpdir, experiment=experiment_name, **test_metrics)

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

    # TODO: Genereate plots over test dataset.
    # 1 -. Error maps
    # 2 -. Histogram comparison
    # 3 -. Predictions vs ground truth vs bilinear
