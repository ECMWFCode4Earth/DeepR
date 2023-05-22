from typing import List

import torch
from torch import nn

from deepr.model.attention import AttentionBlock
from deepr.model.utils import DownSample, ResnetBlock, normalization, swish


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        channel_multipliers: List[int],
        n_resnet_blocks: int,
        in_channels: int,
        z_channels: int
    ):
        """
        Initializes a encoder network for image generation.

        Parameters
        ----------
        channels : int
            The number of feature channels in the first layer.
        channel_multipliers : List[int]
            A list of multipliers used to determine the number of feature channels in each subsequent layer.
        n_resnet_blocks : int
            The number of Resnet blocks per top-level block.
        in_channels : int
            The number of input channels (e.g. 3 for RGB images).
        z_channels : int
            The dimensionality of the embedding space.
        """
        super().__init__()

        # Number of blocks of different resolutions.
        n_resolutions = len(channel_multipliers)
        self.conv_in = nn.Conv2d(in_channels, channels, 3, stride=1, padding=1)
        channels_list = [m * channels for m in [1] + channel_multipliers]

        # List of top-level blocks
        self.down = nn.ModuleList()
        # Create top-level blocks
        for i in range(n_resolutions):
            # Each top level block consists of multiple ResNet Blocks and down-sampling
            resnet_blocks = nn.ModuleList()
            # Add ResNet Blocks
            for _ in range(n_resnet_blocks):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i + 1]))
                channels = channels_list[i + 1]
            # Top-level block
            down = nn.Module()
            down.block = resnet_blocks
            # Down-sampling at the end of each top level block except the last
            if i != n_resolutions - 1:
                down.downsample = DownSample(channels)
            else:
                down.downsample = nn.Identity()
            #
            self.down.append(down)

        # Final ResNet blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttentionBlock(channels)
        self.mid.block_2 = ResnetBlock(channels, channels)

        # Map to embedding space
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, 2 * z_channels, 3, stride=1, padding=1)

    def forward(self, img: torch.Tensor):
        # Map to `channels` with the initial convolution
        x = self.conv_in(img)

        # Top-level blocks
        for down in self.down:
            # ResNet Blocks
            for block in down.block:
                x = block(x)
            # Down-sampling
            x = down.downsample(x)

        # Final ResNet blocks with attention
        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)

        # Normalize and map to embedding space
        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)

        return x
