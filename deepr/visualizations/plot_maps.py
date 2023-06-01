import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_figure_model_samples(
    coarse_image: torch.Tensor,
    fine_image: torch.Tensor,
    prediction: torch.Tensor,
    filename: Optional[str] = None,
):
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

    n_samples = int(coarse_image.shape[0])

    if n_samples != int(fine_image.shape[0]) or n_samples != int(prediction.shape[0]):
        raise ValueError("Inconsistent number of samples between images.")

    figsize = (15, 5 * n_samples)
    fig, axs = plt.subplots(n_samples, 3, figsize=figsize)
    plt.tight_layout()
    if n_samples == 1:
        axs = axs[np.newaxis, ...]
    for i in range(n_samples):
        axs[i, 0].imshow(
            coarse_image[i, 0].numpy()[..., np.newaxis], vmax=vmax, vmin=vmin
        )
        axs[i, 1].imshow(
            fine_image[i, 0].numpy()[..., np.newaxis], vmax=vmax, vmin=vmin
        )
        im = axs[i, 2].imshow(
            prediction[i, 0].numpy()[..., np.newaxis], vmax=vmax, vmin=vmin
        )

        axs[i, 0].get_xaxis().set_visible(False)
        axs[i, 0].get_yaxis().set_visible(False)
        axs[i, 1].get_xaxis().set_visible(False)
        axs[i, 1].get_yaxis().set_visible(False)
        axs[i, 2].get_xaxis().set_visible(False)
        axs[i, 2].get_yaxis().set_visible(False)

        # Titles
        if i == 0:
            axs[i, 0].set_title("ERA5 (Low-res)")
            axs[i, 1].set_title("CERRA (High-res)")
            axs[i, 2].set_title("Prediction (High-res)")

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
        plt.savefig(filename)
        plt.close()

    return fig
