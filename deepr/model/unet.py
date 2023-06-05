from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.unet_2d import UNet2DOutput

from deepr.model.activations import Swish
from deepr.model.unet_blocks import (
    DownBlock,
    Downsample,
    MiddleBlock,
    UpBlock,
    Upsample,
)


class UNet(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        image_channels: int = 1,
        in_channels: int = 1,
        n_channels: int = 16,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        time_embedding_type: str = "positional",
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        channel_multipliers: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attention: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
        n_blocks: int = 2,
    ):
        """
        U-Net.

        NOTE: The spatial shapes of the input must be divisible by 2^{n_resolutions - 1}
        where the number of resolutions is specified by the length of the
        'channel_multipliers' and 'is_attention' arguments.

        Parameters
        ----------
        image_channels : int
            Number of channels in the output image.
        in_channels : int
            Number of channels of the input 2D matrix.
        n_channels : int
            Number of channels in the first layer of the model.
        sample_size : int | Tuple[int, int]
            Spatial dimension of the samples.
        time_embedding_type : str
            Type of time embedding. Options are: "positional" and "fourier".
        freq_shift : int
            Frequency shift of the Fourier time embedding.
        channel_multipliers : Union[Tuple[int, ...], List[int]]
            The channel multiplier for each resolution level of the U-Net.
        is_attention : Union[Tuple[bool, ...], List[int]]
            Whether to use attention mechanism at each resolution level of the U-Net.
        n_blocks : int
            Number of residual blocks at each resolution level of the U-Net.
        conditioned_on_input : Union[bool, int]
            Whether to use conditioning on other inputs, or the number of conditions.
        """
        super().__init__()
        self.sample_size = sample_size
        n_resolutions = len(channel_multipliers)

        # Project input + conditions
        self.image_proj = nn.Conv2d(
            self.config.in_channels,
            n_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
        )

        # Time Embedding
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=n_channels, scale=16
            )
            timestep_input_dim = 2 * n_channels
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(n_channels, flip_sin_to_cos, freq_shift)
            timestep_input_dim = n_channels

        self.time_embedding = TimestepEmbedding(timestep_input_dim, n_channels * 4)

        # First half of U-Net - decreasing resolution
        down: List[nn.Module] = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * channel_multipliers[i]

            # Resnet Blocks
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels, out_channels, n_channels * 4, is_attention[i]
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(
            out_channels,
            n_channels * 4,
        )

        # Second half of U-Net - increasing resolution
        up: List[nn.Module] = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(in_channels, out_channels, n_channels * 4, is_attention[i])
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // channel_multipliers[i]
            up.append(
                UpBlock(in_channels, out_channels, n_channels * 4, is_attention[i])
            )
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(
            in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1)
        )

    def forward(
        self, sample: torch.Tensor, timestep: torch.Tensor, return_dict: bool = True
    ):
        """
        Forward pass.

        Applies the forward pass of the U-Net model on the given input tensor, `sample`,
        and timestep, `timestep`.

        Arguments
        ---------
            sample : torch.Tensor
                The input tensor of the shape (batch_size, num_channels, height, width).
            timestep : torch.Tensor
                The timestep tensor of the shape (batch_size,) representing the timestep
                of each sample.
            return_dict : bool
                Whether or not to return a [`~models.unet_2d.UNet2DOutput`] instead of a
                plain tuple.

        Returns
        -------
            noise: torch.Tensor
                The output tensor of the shape (batch_size, num_classes, height, width).
        """
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.float, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device, dtype=torch.float)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(
            sample.shape[0], dtype=timesteps.dtype, device=timesteps.device
        )

        t_emb = self.time_emb(timesteps).to(dtype=self.dtype)
        t = self.time_embedding(t_emb)

        x = self.image_proj(sample)

        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        x = self.middle(x, t)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        out = self.final(self.act(self.norm(x)))

        if not return_dict:
            return (out,)

        return UNet2DOutput(sample=out)
