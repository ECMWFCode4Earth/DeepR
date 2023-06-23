from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from deepr.utilities.logger import get_logger

logger = get_logger(__name__)


def get_figure_model_samples(
    coarse_image: torch.Tensor,
    fine_image: torch.Tensor,
    prediction: torch.Tensor,
    column_names: List[str] = None,
    filename: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> matplotlib.pyplot.Figure:
    vmax = max(
        float(torch.max(coarse_image)),
        float(torch.max(fine_image)),
        float(torch.max(prediction)),
    )
    vmin = min(
        float(torch.min(coarse_image)),
        float(torch.min(fine_image)),
        float(torch.min(prediction)),
    )
    v_kwargs = {"vmax": vmax, "vmin": vmin}

    n_samples = int(coarse_image.shape[0])

    if n_samples != int(fine_image.shape[0]):
        raise ValueError("Inconsistent number of samples between images.")
    elif int(prediction.shape[0]) % n_samples != 0:
        raise ValueError("Inconsistent number of samples between predictions.")
    else:
        n_realizations = prediction.shape[0] // n_samples

    if figsize is None:
        figsize = (4.5 * (n_realizations + 2), 4.8 * n_samples)
    fig, axs = plt.subplots(n_realizations + 2, n_samples, figsize=figsize)
    plt.tight_layout()
    if n_samples == 1:
        axs = axs[np.newaxis, ...]
    for i in range(n_samples):
        axs[0, i].imshow(coarse_image[i, 0].numpy()[..., np.newaxis], **v_kwargs)
        axs[1, i].imshow(fine_image[i, 0].numpy()[..., np.newaxis], **v_kwargs)
        for r in range(n_realizations):
            im = axs[2 + r, i].imshow(
                prediction[i + r * n_samples, 0].numpy()[..., np.newaxis], **v_kwargs
            )

        axs[0, i].get_xaxis().set_ticks([])
        axs[0, i].get_yaxis().set_ticks([])
        axs[1, i].get_xaxis().set_ticks([])
        axs[1, i].get_yaxis().set_ticks([])

        for r in range(n_realizations):
            axs[2 + r, i].get_xaxis().set_ticks([])
            axs[2 + r, i].get_yaxis().set_ticks([])

        # Titles
        if i == 0:
            axs[0, i].set_ylabel("ERA5 (Low-res)", fontsize=14)
            axs[1, i].set_ylabel("CERRA (High-res)", fontsize=14)
            for r in range(n_realizations):
                axs[2 + r, i].set_ylabel("Prediction (High-res)", fontsize=14)

    if column_names is not None:
        for c, col_name in enumerate(column_names):
            axs[0, c].set_title(col_name, fontsize=14)

    if n_samples == 1:
        fig.subplots_adjust(bottom=0.05)
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
        fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    else:
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax, orientation="vertical")

    if filename is not None:
        logger.info(f"Samples from model have been saved to {filename}")
        plt.savefig(filename, bbox_inches="tight", transparent=True)
        plt.close()

    return fig


def plot_2_maps_comparison(
    matrix1: torch.Tensor,
    matrix2: torch.Tensor,
    matrix_names: List[str] = None,
    metric_name: str = None,
    filename: Optional[str] = None,
    **kwargs,
):
    if "vmax" not in kwargs:
        vmax = max(float(torch.max(matrix1)), float(torch.max(matrix2)))
    else:
        vmax = kwargs["vmax"]

    if "vmin" not in kwargs:
        vmin = min(float(torch.min(matrix1)), float(torch.min(matrix2)))
    else:
        vmin = kwargs["vmin"]

    v_kwargs = {"vmax": vmax, "vmin": vmin}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plt.tight_layout()

    ax1.imshow(matrix1.numpy(), **v_kwargs)
    im = ax2.imshow(matrix2.numpy(), **v_kwargs)

    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])

    if matrix_names is not None:
        ax1.set_title(matrix_names[0].capitalize(), fontsize=18)
        ax2.set_title(matrix_names[1].capitalize(), fontsize=18)

    fig.subplots_adjust(bottom=0.05)
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label=metric_name)

    if filename is not None:
        logger.info(f"Samples from model have been saved to {filename}")
        plt.savefig(filename, bbox_inches="tight", transparent=True)
        plt.close()

    return fig


def plot_2_model_comparison(
    reference: torch.Tensor,
    pred1: torch.Tensor,
    pred2: torch.Tensor,
    matrix_names: List[str] = None,
    metric_name: str = None,
    date: str = None,
    filename: Optional[str] = None,
    **kwargs,
):
    if "vmax" not in kwargs:
        vmax = max(
            float(torch.max(reference)),
            float(torch.max(pred1)),
            float(torch.max(pred2)),
        )
    else:
        vmax = kwargs["vmax"]

    if "vmin" not in kwargs:
        vmin = min(
            float(torch.min(reference)),
            float(torch.min(pred1)),
            float(torch.min(pred2)),
        )
    else:
        vmin = kwargs["vmin"]

    pred_kwargs = {"vmax": vmax, "vmin": vmin, "cmap": "summer"}
    fig = plt.figure(layout="constrained", figsize=(14, 5))
    sfigs = fig.subfigures(1, 2, width_ratios=[1, 2])
    ax = sfigs[0].subplots(1, 1)
    axs = sfigs[1].subplots(2, 2)

    # Reference Matrix
    ax.imshow(reference.numpy(), **pred_kwargs)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    # Prediction and errors
    error1 = pred1 - reference
    error2 = pred2 - reference

    emax = max(float(torch.max(torch.abs(error1))), float(torch.max(torch.abs(error2))))

    axs[0, 0].imshow(pred1.numpy(), **pred_kwargs)
    axs[1, 0].imshow(error1.numpy(), vmin=-emax, vmax=emax, cmap="RdBu")
    im = axs[0, 1].imshow(pred2.numpy(), **pred_kwargs)
    im_e = axs[1, 1].imshow(error2.numpy(), vmin=-emax, vmax=emax, cmap="RdBu")

    for ax_unraveled in axs.ravel():
        ax_unraveled.get_xaxis().set_ticks([])
        ax_unraveled.get_yaxis().set_ticks([])

    if date is not None:
        ax.set_xlabel(date, fontsize=18)

    if matrix_names is not None:
        ax.set_title(matrix_names[0], fontsize=18)
        axs[0, 0].set_title(matrix_names[1], fontsize=18)
        axs[0, 1].set_title(matrix_names[2], fontsize=18)

    sfigs[1].colorbar(
        im, ax=axs[0, 1], orientation="vertical", label=f"Prediction ({metric_name})"
    )
    sfigs[1].colorbar(
        im_e, ax=axs[1, 1], orientation="vertical", label=f"Abs. Error ({metric_name})"
    )

    if filename is not None:
        logger.info(f"Samples from model have been saved to {filename}")
        plt.savefig(filename, bbox_inches="tight", transparent=False)
        plt.close()

    return fig
