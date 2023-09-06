from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy
import torch

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
) -> plt.Figure:
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
    plt.Figure
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
        -9999.0 if baseline is None else float(torch.max(baseline)),
    )
    vmin = min(
        float(torch.min(fine_image)),
        float(torch.min(prediction)),
        9999.0 if input_image is None else float(torch.min(input_image)),
        9999.0 if baseline is None else float(torch.min(baseline)),
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
    if n_samples > 1:
        fig, axs = plt.subplots(n_realizations + n_extras, n_samples, figsize=fig_size)
        func_name = "set_ylabel"
    elif n_samples == 1:  # if only one row, it is necessary to include in the axes
        fig, axs = plt.subplots(1, n_realizations + n_extras, figsize=fig_size)
        axs = axs[..., numpy.newaxis]
        func_name = "set_title"
    plt.tight_layout()

    # Loop over the number of columns, which is the same as the number of samples
    for i in range(n_samples):
        if input_image is not None:
            axs[0, i].imshow(input_image[i, 0].numpy()[..., numpy.newaxis], **v_kwargs)
            axs[1, i].imshow(fine_image[i, 0].numpy()[..., numpy.newaxis], **v_kwargs)
        else:
            axs[0, i].imshow(fine_image[i, 0].numpy()[..., numpy.newaxis], **v_kwargs)

        # Loop over the number of rows in the column
        for r in range(n_realizations):
            # Plot the predictions
            im = axs[n_extras + r, i].imshow(
                prediction[i + r * n_samples, 0].numpy()[..., numpy.newaxis], **v_kwargs
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
                getattr(axs[0, i], func_name)("ERA5 (Low-res)", fontsize=14)
                getattr(axs[1, i], func_name)("CERRA (High-res)", fontsize=14)
            else:
                getattr(axs[0, i], func_name)("CERRA (High-res)", fontsize=14)

            for r in range(n_realizations):
                if baseline is not None and r == n_realizations - 1:
                    label = "Bicubic Int."
                else:
                    label = "Prediction (High-res)"
                getattr(axs[n_extras + r, i], func_name)(label, fontsize=14)

    # Title of the columns
    if column_names is not None:
        for c, col_name in enumerate(column_names):
            axs[0, c].set_title(col_name, fontsize=14)

    # Include the color bar in the depiction
    if n_samples == 1:
        fig.subplots_adjust(bottom=0.05)
        cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.05])
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
