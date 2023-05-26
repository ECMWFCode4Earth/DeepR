from typing import List, Tuple, Union

import torch
import torch.nn as nn

from deepr.model.activations import Swish
from deepr.model.attention import AttentionBlock
from deepr.model.resnet import ResidualBlock
from deepr.model.utils import Downsample, TimeEmbedding, Upsample


class DownBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool
    ):
        """Downsampling block class.

        These are used in the first half of U-Net at each resolution.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        time_channels : int
            The number of time channels.
        has_attn : bool
            A flag indicating whether to use attention block or not.
        """
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool
    ):
        """Upsampling block class.

        These are used in the second half of U-Net at each resolution.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        time_channels : int
            The number of time channels.
        has_attn : bool
            A flag indicating whether to use attention block or not.
        """
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output
        # of the same resolution from the first half of the U-Net
        self.res = ResidualBlock(
            in_channels + out_channels, out_channels, time_channels
        )
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        n_channels: int = 16,
        channel_multipliers: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attention: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
        n_blocks: int = 2,
    ):
        super().__init__()
        n_resolutions = len(channel_multipliers)
        self.image_proj = nn.Conv2d(
            image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1)
        )
        self.time_emb = TimeEmbedding(n_channels * 4)

        # First half of U-Net - decreasing resolution
        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * channel_multipliers[i]
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
        up = []
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

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = self.time_emb(t)
        x = self.image_proj(x)

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

        return self.final(self.act(self.norm(x)))
