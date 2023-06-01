import torch
from torch import nn

from deepr.model.activations import Swish


class ResidualBlock(nn.Module):
    """
    Residual block.

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        n_groups: int = 8,
        dropout: float = 0.1,
    ):
        """CNN block with Group Normalization, Swish activation, and Conv. layers.

        The block takes input channel values, output channel values, time channels,
        number of groups (n_groups), and dropout rate as parameters.

        Parameters
        ----------
        in_channels: int
            Number of input channels.
        out_channels: int
            Number of output channels.
        time_channels: int
            Number of time channels.
        n_groups: int, optional (default=`32`)
            Number of groups.
        dropout: float, optional (default=`0.1`)
            Dropout rate.
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # If the number of input channels is not equal to the number of output channels
        # we have to project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
            x : torch.Tensor
                Input vector with shape `[batch_size, in_channels, height, width]`.
            t : torch.Tensor
                Time vector `[batch_size, time_channels]`.

        Returns
        -------
            torch.Tensor: vector with shape `[batch_size, out_channels, height, width]`.
        """
        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_emb(self.time_act(t))[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        if hasattr(self, "shortcut"):
            h += self.shortcut(x)
        return h
