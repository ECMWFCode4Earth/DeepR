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
from deepr.visualizations.plot_rose import plot_rose

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

    maps_cfg = config["visualizations"].get("metrics_global_map", None)
    # Obtain error maps
    if maps_cfg is not None:
        (
            mae,
            mse,
            r2,
            mae_base,
            mse_base,
            r2_base,
            improvement,
        ) = compute_model_and_baseline_errors(
            pipe,
            dataloader,
            baseline=config["baseline"],
            scaler_func=scaler_func,
            inference_steps=config["inference_steps"],
            num_batches=maps_cfg["num_batches"],
        )

        # Compute error maps to compare spatial metric by hour (and for all the hours)
        names = [pipe.__class__.__name__, config["baseline"]]
        if maps_cfg.get("map_plots", True):
            visualization_local_dir = f"{local_dir}/plot_2_maps_comparison"
            os.makedirs(visualization_local_dir, exist_ok=True)
            for time_value in [0, 3, 6, 9, 12, 15, 18, 21, "all"]:
                plot_2_maps_comparison(
                    mse[time_value],
                    mse_base[time_value],
                    names,
                    "MSE (ºC)",
                    f"{visualization_local_dir}/mse_vs_{config['baseline']}_{time_value}.png",
                    vmin=0,
                )
                plot_2_maps_comparison(
                    mae[time_value],
                    mae_base[time_value],
                    names,
                    "MAE (ºC)",
                    f"{visualization_local_dir}/mae_vs_{config['baseline']}_{time_value}.png",
                    vmin=0,
                )
                plot_2_maps_comparison(
                    r2[time_value],
                    r2_base[time_value],
                    names,
                    "R2",
                    f"{visualization_local_dir}/r2_vs_{config['baseline']}_{time_value}.png",
                    vmin=-1,
                )

        # Compute rose plot to compare total metric by hour (and for all the hours)
        if maps_cfg.get("rose_plot", True):
            visualization_local_dir = f"{local_dir}/rose-plot"
            os.makedirs(visualization_local_dir, exist_ok=True)
            colors = ["#390099", "#9e0059"]
            plot_rose(
                {key: value for key, value in mae.items() if key != "all"},
                {key: value for key, value in mae_base.items() if key != "all"},
                None,
                names=names,
                custom_colors=colors,
                title="MAE (ºC)",
                output_path=f"{visualization_local_dir}/rose-plot_mae.png",
            )
            plot_rose(
                {key: value for key, value in mse.items() if key != "all"},
                {key: value for key, value in mse_base.items() if key != "all"},
                None,
                names=names,
                custom_colors=colors,
                title="MSE (ºC)",
                output_path=f"{visualization_local_dir}/rose-plot_mse.png",
            )
            land_mask_array = dataset.add_auxiliary_features["lsm-high"].lsm.as_numpy().values
            plot_rose(
                {key: value for key, value in mae.items() if key != "all"},
                {key: value for key, value in mae_base.items() if key != "all"},
                ("land", land_mask_array),
                names=names,
                custom_colors=colors,
                title="MAE (ºC) - only land points",
                output_path=f"{visualization_local_dir}/rose-plot_mae-on-land.png",
            )
            plot_rose(
                {key: value for key, value in mse.items() if key != "all"},
                {key: value for key, value in mse_base.items() if key != "all"},
                ("land", land_mask_array),
                names=names,
                custom_colors=colors,
                title="MSE (ºC) - only land points",
                output_path=f"{visualization_local_dir}/rose-plot_mse-on-land.png",
            )
            plot_rose(
                {key: value for key, value in mae.items() if key != "all"},
                {key: value for key, value in mae_base.items() if key != "all"},
                ("sea", land_mask_array),
                names=names,
                custom_colors=colors,
                title="MAE (ºC) - only sea points",
                output_path=f"{visualization_local_dir}/rose-plot_mae-on-sea.png",
            )
            plot_rose(
                {key: value for key, value in mse.items() if key != "all"},
                {key: value for key, value in mse_base.items() if key != "all"},
                ("sea", land_mask_array),
                names=names,
                custom_colors=colors,
                title="MSE (ºC) - only sea points",
                output_path=f"{visualization_local_dir}/rose-plot_mse-on-sea.png",
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
