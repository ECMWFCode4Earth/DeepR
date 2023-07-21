from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from deepr.utilities.logger import get_logger

logger = get_logger(__name__)


def get_figure_model_samples(
    fine_image: torch.Tensor,
    prediction: torch.Tensor,
    input_image: torch.Tensor = None,
    baseline: torch.Tensor = None,
    column_names: List[str] = None,
    filename: Optional[str] = None,
    fig_size: Optional[Tuple[int, int]] = None,
) -> matplotlib.pyplot.Figure:
    """
    Generate a figure displaying model samples.

    Parameters
    ----------
    fine_image : torch.Tensor
        Fine-resolution images.
    prediction : torch.Tensor
        Model predictions.
    input_image : torch.Tensor, optional
        Low-resolution input images. Default is None.
    baseline : torch.Tensor, optional
        Baseline images. Default is None.
    column_names : List[str], optional
        Names for columns in the figure. Default is None.
    filename : str, optional
        Filename to save the figure. Default is None.
    fig_size : Tuple[int, int], optional
        Figure size (width, height) in inches. Default is None.

    Returns
    -------
    matplotlib.pyplot.Figure
        The generated figure.
    """
    # Concatenating baseline to predictions, if baseline exists
    if baseline is not None:
        prediction = torch.cat([prediction, baseline], dim=0)

    n_extras = 2 if input_image is not None else 1

    # Defining the maximum and minimum value of the data that will be depicted
    vmax = max(
        float(torch.max(fine_image)),
        float(torch.max(prediction)),
        -9999.0 if input_image is None else float(torch.max(input_image)),
        -9999.0 if input_image is None else float(torch.max(baseline)),
    )
    vmin = min(
        float(torch.min(fine_image)),
        float(torch.min(prediction)),
        9999.0 if input_image is None else float(torch.min(input_image)),
        9999.0 if input_image is None else float(torch.min(baseline)),
    )
    v_kwargs = {"vmax": vmax, "vmin": vmin}

    n_samples = int(fine_image.shape[0])

    if input_image is not None and n_samples != int(input_image.shape[0]):
        raise ValueError("Inconsistent number of samples between images.")
    elif int(prediction.shape[0]) % n_samples != 0:
        raise ValueError("Inconsistent number of samples between predictions.")
    else:
        n_realizations = prediction.shape[0] // n_samples

    # Defining the figure size if it is not provided as argument
    if fig_size is None:
        fig_size = (4.5 * (n_realizations + n_extras), 4.8 * n_samples)

    # Defining the figure and axes
    fig, axs = plt.subplots(n_realizations + n_extras, n_samples, figsize=fig_size)
    if n_samples == 1:  # if only one row, it is necessary to include in the axes
        axs = axs[..., np.newaxis]
    plt.tight_layout()

    # Loop over the number of columns, which is the same as the number of samples
    for i in range(n_samples):
        if input_image is not None:
            axs[0, i].imshow(input_image[i, 0].numpy()[..., np.newaxis], **v_kwargs)
            axs[1, i].imshow(fine_image[i, 0].numpy()[..., np.newaxis], **v_kwargs)
        else:
            axs[0, i].imshow(fine_image[i, 0].numpy()[..., np.newaxis], **v_kwargs)

        # Loop over the number of rows in the column
        for r in range(n_realizations):
            # Plot the predictions
            im = axs[n_extras + r, i].imshow(
                prediction[i + r * n_samples, 0].numpy()[..., np.newaxis], **v_kwargs
            )

        axs[0, i].get_xaxis().set_ticks([])
        axs[0, i].get_yaxis().set_ticks([])
        axs[1, i].get_xaxis().set_ticks([])
        axs[1, i].get_yaxis().set_ticks([])

        for r in range(n_realizations):
            axs[n_extras + r, i].get_xaxis().set_ticks([])
            axs[n_extras + r, i].get_yaxis().set_ticks([])

        # Title of the rows
        if i == 0:
            if input_image is not None:
                axs[0, i].set_ylabel("ERA5 (Low-res)", fontsize=14)
                axs[1, i].set_ylabel("CERRA (High-res)", fontsize=14)
            else:
                axs[0, i].set_ylabel("CERRA (High-res)", fontsize=14)

            for r in range(n_realizations):
                if baseline is not None and r == n_realizations - 1:
                    label = "Bicubic Int."
                else:
                    label = "Prediction (High-res)"
                axs[n_extras + r, i].set_ylabel(label, fontsize=14)

    # Title of the columns
    if column_names is not None:
        for c, col_name in enumerate(column_names):
            axs[0, c].set_title(col_name, fontsize=14)

    # Include the color bar in the depiction
    if n_samples == 1:
        fig.subplots_adjust(bottom=0.05)
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
        fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    else:
        fig.subplots_adjust(right=0.95)
        cbar_ax = fig.add_axes([0.97, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax, orientation="vertical")

    # Save figure if the name of a file is provided
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


def plot_simple_map(
    data,
    vmin=None,
    vmax=None,
    cmap: str = None,
    label: str = "Temperature (ºC)",
    out_file: str = None,
):
    if cmap is None and vmin * vmax < 0:  # opposite signs
        colors = [(0, "blue"), (-vmin / (vmax - vmin), "white"), (1, "red")]
        cmap = LinearSegmentedColormap.from_list("temp", colors)
    plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.axis("off")
    if cmap is not None:
        plt.colorbar(shrink=0.65, label=label)
    if out_file is not None:
        plt.savefig(out_file, transparent=True, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        return plt.clf()
