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
from deepr.validation.sample_predictions import (
    sample_diffusion_samples_random,
    sample_gif,
)
from deepr.visualizations.plot_maps import plot_2_maps_comparison

tmpdir = tempfile.mkdtemp(prefix="test-")

experiment_name = "Test Neural Network"


def validate_model(
    model,
    scheduler,
    dataset: torch.utils.data.IterableDataset,
    config: dict,
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
    """
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset, config["batch_size"], pin_memory=True
    )

    # Instantiate pipeline
    pipe = cDDPMPipeline(unet=model, scheduler=scheduler, obs_model=None).to(
        config["device"]
    )

    # Define scaler function if label scaler is provided
    scaler_func = None if label_scaler is None else label_scaler.apply_inverse_scaler

    # Define local directory for saving evaluation results
    local_dir = f"{config['output_directory']}/hf-{model.__class__.__name__}-evaluation"
    os.makedirs(name=local_dir, exist_ok=True)

    # Clone Hugging Face repository if provided
    if config["push_to_hub"] and hf_repo_name is not None:
        repo = Repository(
            local_dir, clone_from=hf_repo_name, token=os.getenv("HF_TOKEN")
        )
        repo.git_pull()

    # Sample GIFF predictions
    giff_sample_cfg = config["visualizations"].get("giff_timestep_freq", None)
    if giff_sample_cfg is not None:
        odir = local_dir + "/animated_diffusion"
        os.makedirs(odir, exist_ok=True)
        sample_gif(
            pipe,
            dataloader,
            scaler_func=scaler_func,
            output_dir=odir,
            inference_steps=config["inference_steps"],
            fps=giff_sample_cfg["fps"],
            freq_timesteps_frame=giff_sample_cfg["freq_timesteps"],
        )

    # Show samples compared with other models
    samples_cfg = config["visualizations"].get("sample_observation_vs_prediction", None)
    if samples_cfg is not None:
        odir = local_dir + "/samples_comparison"
        os.makedirs(odir, exist_ok=True)        
        sample_diffusion_samples_random(
            pipe,
            dataloader,
            scaler_func=scaler_func,
            baseline=config["baseline"],
            output_dir=odir,
            num_samples=samples_cfg["num_samples"],
            num_realizations=samples_cfg["num_realizations"],
            inference_steps=config["inference_steps"],
            device=config["device"],
        )

    # Obtain error maps
    mae, mse, mae_base, mse_base, improvement = compute_model_and_baseline_errors(
        pipe, dataloader, config["baseline"], scaler_func
    )
    names = [pipe.__class__.__name__, config["baseline"]]
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
