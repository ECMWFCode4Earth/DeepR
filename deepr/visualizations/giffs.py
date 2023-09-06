import tempfile
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from deepr.visualizations.plot_maps import plot_simple_map


def generate_giff(
    latents: torch.Tensor, filename: str, fps: int = 50, label: str = "Temperature"
):
    vmin = torch.min(latents[-1, ...])
    vmax = torch.max(latents[-1, ...])

    with tempfile.TemporaryDirectory(suffix="-giff-diffusion") as f:
        fig_paths = []
        for t in tqdm(range(latents.shape[0]), desc="Plotting frames for GIFF"):
            fname = Path(f) / f"latents_{t}.png"
            plot_simple_map(latents[t], vmin, vmax, label=label, out_file=fname)
            fig_paths.append(fname)

        imgs = [Image.open(f) for f in fig_paths]
    if not filename.endswith(".gif"):
        filename += ".gif"

    imgs[0].save(
        fp=filename,
        format="GIF",
        append_images=imgs,
        save_all=True,
        optimize=True,
        duration=max(20, int(1e3 / fps)),  # 1 frame each 20ms = 50 fps (min value)
        loop=3,
    )
    return filename
