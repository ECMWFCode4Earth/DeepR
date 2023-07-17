import os
import tempfile

import evaluate
import torch
from huggingface_hub import Repository

from deepr.data.scaler import XarrayStandardScaler
from deepr.model.conditional_ddpm import cDDPMPipeline
from deepr.validation.nn_performance_metrics import (
    compute_and_upload_metrics,
    compute_model_and_baseline_errors,
)
from deepr.validation.sample_predictions import sample_observation_vs_prediction
from deepr.visualizations.plot_maps import plot_2_maps_comparison

tmpdir = tempfile.mkdtemp(prefix="test-")

experiment_name = "Test Neural Network"


def validate_model(
    model,
    scheduler,
    dataset: torch.utils.data.IterableDataset,
    config: dict,
    batch_size: int = os.getenv("BATCH_SIZE", 4),
    hf_repo_name: str = None,
    label_scaler: XarrayStandardScaler = None,
    push_to_hub: bool = False,
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

    # Instantiate pipeline
    cDDPMPipeline(unet=model, scheduler=scheduler, obs_model=None)

    # Define scaler function if label scaler is provided
    scaler_func = None if label_scaler is None else label_scaler.apply_inverse_scaler

    # Define local directory for saving evaluation results
    local_dir = f"{config['output_directory']}/hf-{model.__class__.__name__}-evaluation"
    os.makedirs(name=local_dir, exist_ok=True)

    # Clone Hugging Face repository if provided
    if push_to_hub and hf_repo_name is not None:
        repo = Repository(
            local_dir, clone_from=hf_repo_name, token=os.getenv("HF_TOKEN")
        )
        repo.git_pull()

    # Show samples compared with other models
    if config["visualizations"]["sample_observation_versus_prediction"] > 0:
        visualization_local_dir = f"{local_dir}/sample_observation_versus_prediction"
        os.makedirs(visualization_local_dir, exist_ok=True)
        sample_observation_vs_prediction(
            model,
            dataloader,
            visualization_local_dir,
            scaler_func,
            config["baseline"],
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
