import torch
from torch import nn

from deepr.model.attention import AttentionBlock
from deepr.model.resnet import ResidualBlock


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention
        # layer function signature to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)


class DownBlock(nn.Module):
    """Down Block class.

    It represents a block in the first half of U-Net where the input features are being
    encoded.

    Attributes
    ----------
    res : ResidualBlock
        A residual block.
    final_layer : Type[nn.Module]
        The final layer after the Residual Block. If has_attn is True, it is
        `deepr.model.attention.AttentionBlock`. Otherwise it is `nn.Identity`.
    """

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
        self.final_layer = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.final_layer(x)
        return x


class UpBlock(nn.Module):
    """Up Block class.

    It represents a block in the second half of U-Net where the input features are being
    decoded.

    Attributes
    ----------
    res : ResidualBlock
        A residual block.
    final_layer : Type[nn.Module]
        The final layer after the Residual Block. If has_attn is True, it is
        `deepr.model.attention.AttentionBlock`. Otherwise it is `nn.Identity`.
    """

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
        self.final_layer = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.final_layer(x)
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
