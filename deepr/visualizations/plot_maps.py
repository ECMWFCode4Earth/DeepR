import logging
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_figure_model_samples(
    coarse_image: torch.Tensor,
    fine_image: torch.Tensor,
    prediction: torch.Tensor,
    column_names: List[str] = None,
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
    v_kwargs = {"vmax": vmax, "vmin": vmin}

    n_samples = int(coarse_image.shape[0])

    if n_samples != int(fine_image.shape[0]):
        raise ValueError("Inconsistent number of samples between images.")
    elif int(prediction.shape[0]) % n_samples != 0:
        raise ValueError("Inconsistent number of samples between predictions.")
    else:
        n_realizations = prediction.shape[0] // n_samples

    figsize = (7.7 * n_realizations, 4.8 * n_samples)
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
        plt.savefig(filename)
        plt.close()

    return fig
