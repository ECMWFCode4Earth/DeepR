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
        out_channels: int = 1,
        in_channels: int = 1,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        time_embedding_type: str = "positional",
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        block_out_channels: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attention: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
        layers_per_block: int = 2,
    ):
        """
        U-Net.

        NOTE: The spatial shapes of the input must be divisible by 2^{n_resolutions - 1}
        where the number of resolutions is specified by the length of the
        'channel_multipliers' and 'is_attention' arguments.

        Parameters
        ----------
        out_channels : int
            Number of channels in the output image.
        in_channels : int
            Number of channels of the input 2D matrix.
        sample_size : int | Tuple[int, int]
            Spatial dimension of the samples.
        time_embedding_type : str
            Type of time embedding. Options are: "positional" and "fourier".
        freq_shift : int
            Frequency shift of the Fourier time embedding.
        block_out_channels : Union[Tuple[int, ...], List[int]]
            The output channels for each resolution level of the U-Net.
        is_attention : Union[Tuple[bool, ...], List[int]]
            Whether to use attention mechanism at each resolution level of the U-Net.
        layers_per_block : int
            Number of residual blocks at each resolution level of the U-Net.
        conditioned_on_input : Union[bool, int]
            Whether to use conditioning on other inputs, or the number of conditions.
        """
        super().__init__()
        self.sample_size = sample_size
        n_resolutions = len(block_out_channels)
        init_channels = block_out_channels[0]

        # Project input + conditions
        self.image_proj = nn.Conv2d(
            self.config.in_channels,
            init_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
        )

        # Time Embedding
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=init_channels, scale=16
            )
            timestep_input_dim = 2 * init_channels
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(init_channels, flip_sin_to_cos, freq_shift)
            timestep_input_dim = init_channels

        self.time_embedding = TimestepEmbedding(timestep_input_dim, init_channels * 4)

        # First half of U-Net - decreasing resolution
        down: List[nn.Module] = []
        in_ch_down = init_channels
        for i, out_ch_down in enumerate(block_out_channels):
            # Resnet Blocks
            for _ in range(layers_per_block):
                down.append(
                    DownBlock(
                        in_ch_down, out_ch_down, init_channels * 4, is_attention[i]
                    )
                )
                in_ch_down = out_ch_down
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_ch_down))

        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_ch_down, init_channels * 4)
        in_ch_up = out_ch_down

        # Second half of U-Net - increasing resolution
        up: List[nn.Module] = []
        for i, out_ch_up in reversed(list(enumerate(block_out_channels))):
            for _ in range(layers_per_block):
                up.append(
                    UpBlock(in_ch_up, in_ch_up, init_channels * 4, is_attention[i])
                )

            # Final block to reduce the number of channels
            up.append(UpBlock(in_ch_up, out_ch_up, init_channels * 4, is_attention[i]))
            in_ch_up = out_ch_up

            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_ch_up))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, init_channels)
        self.act = Swish()
        self.final = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
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

        t_emb = self.time_proj(timesteps).to(dtype=self.dtype)
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
