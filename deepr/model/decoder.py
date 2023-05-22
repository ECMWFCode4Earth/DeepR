from typing import List

import torch
from torch import nn

from deepr.model.attention import AttentionBlock
from deepr.model.utils import ResnetBlock, UpSample, normalization, swish


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        channel_multipliers: List[int],
        n_resnet_blocks: int,
        out_channels: int,
        z_channels: int
    ):
        """
        Initialize a Multi-Resolution Attention Network.

        Parameters
        ----------
        channels : int
            Number of channels in the top-level block.
        channel_multipliers : List[int]
            List of channel multipliers for each resolution. The length of the list is
            the number of resolutions, from low to high.
        n_resnet_blocks : int
            Number of ResNet blocks in each top-level block.
        out_channels : int
            Number of output channels.
        z_channels : int
            Number of channels in the embedding space.
        """
        super().__init__()

        num_resolutions = len(channel_multipliers)
        channels_list = [m * channels for m in channel_multipliers]
        channels = channels_list[-1]
        self.conv_in = nn.Conv2d(z_channels, channels, 3, stride=1, padding=1)

        # ResNet blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttentionBlock(channels)
        self.mid.block_2 = ResnetBlock(channels, channels)

        # List of top-level blocks
        self.up = nn.ModuleList()
        for i in reversed(range(num_resolutions)):
            # Each top level block consists of multiple ResNet Blocks and up-sampling
            resnet_blocks = nn.ModuleList()

            # Add ResNet Blocks
            for _ in range(n_resnet_blocks + 1):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i]))
                channels = channels_list[i]

            # Top-level block
            up = nn.Module()
            up.block = resnet_blocks

            # Up-sampling at the end of each top level block except the first
            if i != 0:
                up.upsample = UpSample(channels)
            else:
                up.upsample = nn.Identity()
            self.up.insert(0, up)

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, out_channels, 3, stride=1, padding=1)

    def forward(self, z: torch.Tensor):
        h = self.conv_in(z)

        # ResNet blocks with attention
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Top-level blocks
        for up in reversed(self.up):
            # ResNet Blocks
            for block in up.block:
                h = block(h)
            # Up-sampling
            h = up.upsample(h)

        # Normalize and map to image space
        h = self.norm_out(h)
        h = swish(h)
        img = self.conv_out(h)

        return img
