import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from deepr.model.conditional_ddpm import cDDPMPipeline
from deepr.model.utils import get_hour_embedding
from deepr.visualizations.plot_maps import plot_2_model_comparison, plot_simple_map
from deepr.visualizations.plot_samples import get_figure_model_samples

K_to_C = 273.15
logger = logging.getLogger(__name__)
tmpdir = Path(tempfile.mkdtemp())


def sample_observation_vs_prediction(
    model,
    dataloader: torch.utils.data.DataLoader,
    local_dir: str,
    scaler_func: Callable = None,
    baseline: str = "bicubic",
    num_samples: int = 10,
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
    num_samples : int, optional
        The number of samples to randomly select and compare, by default 10.
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

        for i in range(len(times)):
            if random.choice([True, False]):
                continue
            filename = Path(local_dir) / f"pred_comparison_{samples_get}.png"
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
            if samples_get == num_samples:
                return None


def sample_diffusion_samples_random(
    pipeline: cDDPMPipeline,
    dataloader: torch.utils.data.DataLoader,
    scaler_func: Callable = None,
    baseline: str = "bicubic",
    num_samples: int = 10,
    num_realizations: int = 3,
    inference_steps: int = 1000,
    output_dir: str = None,
    device: str = "",
):
    n_samples = 0
    for i, (era5, cerra, times) in enumerate(dataloader):
        # Prepare data
        # 1 A) Encode hour
        hour_emb = get_hour_embedding(times[:, :1], "class", 24)

        # 1 B) Repeat each sample by number of realizations
        era5_repeated = era5.repeat(num_realizations, 1, 1, 1)

        if hour_emb is not None:
            hour_emb = hour_emb.to(device)
            hour_emb = hour_emb.repeat(num_realizations, 1).squeeze()

        # 1 C) Compute baseline predictions
        pred_base = torch.nn.functional.interpolate(
            era5[..., 6:-6, 6:-6], scale_factor=5, mode=baseline
        )

        # 2) Run the predictions
        pred_nn = pipeline(
            images=era5_repeated,
            class_labels=hour_emb,
            num_inference_steps=inference_steps,
            generator=torch.manual_seed(2023),
            output_type="tensor",
        ).images

        if scaler_func is not None:
            cerra = scaler_func(cerra, times[:, 2]) - K_to_C
            pred_nn = (
                scaler_func(pred_nn, times[:, 2].repeat(num_realizations)) - K_to_C
            )
            pred_base = scaler_func(pred_base, times[:, 2]) - K_to_C

        # Make a grid out of the images
        sample_names = [f"{t[0]:d}H {t[1]:02d}-{t[2]:02d}-{t[3]:04d}" for t in times]
        get_figure_model_samples(
            cerra.cpu(),
            pred_nn.cpu(),
            # input_image=era5.cpu(),
            baseline=pred_base.cpu(),
            column_names=sample_names,
            filename=output_dir + f"/samples_{i}.png",
        )
        n_samples += len(sample_names)
        if n_samples >= num_samples:
            return None
        


def sample_gif(
    pipeline,
    dataloader,
    scaler_func: Callable = None,
    output_dir: str = None,
    freq_timesteps_frame: int = 1,
    inference_steps: int = 1000,
    fps: int = 50,
):
    """
    Generate GIFs of the diffusion process for a given pipeline.

    Args:
    ----
        pipeline (callable): The pipeline function to apply to the images.
        dataloader (iterable): An iterable containing low-resolution reanalysis.
        scaler_func (callable, optional): A function to un-scale the images. Defaults to None.
        output_dir (str, optional): The directory to save the generated GIFs. Defaults to None.
        freq_timesteps_frame (int, optional): The frequency of diffusion timesteps to
            save as frames in the GIFs. Defaults to 1, which saves latents at all
            timesteps as frames.
        inference_steps (int, optional): The number of inference timesteps to perform
            the diffusion process. Defaults to 1000.
        fps (int, optional): The frames per second to show. Maximum value supported for
            most of modern browsers is 50fps.
    """
    era5, _, times = next(iter(dataloader))
    hr_im, interm = pipeline(
        images=era5,
        class_labels=times[:, :1],
        generator=torch.manual_seed(2023),
        num_inference_steps=60,
        return_dict=False,
        saving_freq_interm=freq_timesteps_frame,
        output_type="tensor",
    )
    if scaler_func is not None:
        hr_im = scaler_func(hr_im, times[:, 2])

    # Generate GIFFS
    for i, time in enumerate(times):
        date = f"{time[1]:02d}/{time[2]:02d}/{time[3]:04d}"
        logger.info(f"Generating GIF for time: {date}")
        vmin = torch.min(hr_im[i, ...]) - K_to_C
        vmax = torch.max(hr_im[i, ...]) - K_to_C

        odir = tmpdir / date
        os.makedirs(odir, exist_ok=True)
        fig_paths = []
        for t in range(interm.shape[1]):
            fname = odir / f"{t}.png"
            if scaler_func is not None:
                im = (
                    scaler_func(
                        interm[i : i + 1, t : t + 1, ...], times[i : i + 1, 2]
                    ).squeeze()
                    - K_to_C
                )
            else:
                im = interm[i, t] - K_to_C
            plot_simple_map(im, vmin, vmax, "autumn", "Temperature (ºC)", fname)
            fig_paths.append(fname)

        imgs = [Image.open(f) for f in fig_paths]
        imgs[0].save(
            fp=output_dir
            + f"/diffusion_{time[0]:d}H-{date}_{inference_steps}steps.gif",
            format="GIF",
            append_images=imgs,
            save_all=True,
            optimize=True,
            duration=max(20, int(1e3 / fps)),  # 1 frame each 20ms = 50 fps (min value)
            loop=0,
        )
        del imgs
