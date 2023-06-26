import logging
from math import ceil, log2
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

logger = logging.getLogger(__name__)


class ConvBaselineConfig(PretrainedConfig):
    model_type = "convbilinear"

    attribute_map = {"hidden_size": "embed_dim"}

    def __init__(
        self,
        num_channels: int = 1,
        upscale: int = 1,
        interpolation_method: str = "bicubic",
        input_shape: Tuple[int] = None,
        image_size: Tuple[int] = None,
        upblock_channels: List[int] = [64, 32],
        upblock_kernel_size: List[int] = [5, 3],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.upscale = upscale
        self.input_shape = input_shape
        self.image_size = tuple(map(lambda x: x // self.upscale, self.sample_size))
        self.upblock_channels = upblock_channels
        self.upblock_kernel_size = upblock_kernel_size
        self.interpolation_method = interpolation_method
        self.upscale_power2 = int(ceil(log2(upscale)))


class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        interm_channels: List[int] = [64, 32],
        kernel_size: List[int] = [5, 3],
        upscale_ratio: int = 2,
    ):
        super(UpConvBlock, self).__init__()
        n_channels = out_channel * (upscale_ratio**2)

        self.conv1 = nn.Conv2d(
            in_channel, interm_channels[0], kernel_size=5, padding="same"
        )
        self.conv2 = nn.Conv2d(
            interm_channels[0], interm_channels[1], kernel_size=3, padding="same"
        )
        self.conv3 = nn.Conv2d(
            interm_channels[1], n_channels, kernel_size=1, padding="same"
        )
        self.conv4 = nn.ConvTranspose2d(
            n_channels, out_channel, kernel_size=2, padding=0, stride=upscale_ratio
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        return out


class ConvBaseline(PreTrainedModel):
    config_class = ConvBaselineConfig
    base_model_prefix = "convbaseline"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    upscale_ratio_upconv = 2

    def __init__(self, config: ConvBaselineConfig):
        super().__init__(config)

        self.input_upconv_shape = np.array(config.sample_size) / (
            self.upscale_ratio_upconv**self.config.upscale_power2
        )
        kernel_size = (
            np.array(config.input_shape) - self.input_upconv_shape + 1
        ).astype(int)
        extra_pixels = np.array(config.input_shape) - config.image_size
        self.from_lat = int(extra_pixels[0] // 2)
        self.from_lon = int(extra_pixels[1] // 2)
        self.to_lat = -self.from_lat if self.from_lat > 0 else None
        self.to_lon = -self.from_lon if self.from_lon > 0 else None

        self.preprocess_model = nn.Conv2d(
            config.num_channels,
            config.num_channels * self.upscale_ratio_upconv**2,
            kernel_size=tuple(kernel_size),
            padding=0,
        )

        convs: List[nn.Module] = []
        in_channels = config.num_channels * self.upscale_ratio_upconv**2
        for i in range(self.config.upscale_power2):
            j = self.config.upscale_power2 - i - 1
            out_channel = config.num_channels * (self.upscale_ratio_upconv**2) ** j
            convs.append(
                UpConvBlock(
                    in_channel=in_channels,
                    out_channel=out_channel,
                    interm_channels=config.upblock_channels,
                    kernel_size=config.upblock_kernel_size,
                    upscale_ratio=self.upscale_ratio_upconv,
                )
            )
            in_channels = out_channel
        self.cnns = nn.ModuleList(convs)

        super().post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        # Baseline interpoletion
        out_baseline = torch.nn.functional.interpolate(
            pixel_values[..., self.from_lat : self.to_lat, self.from_lon : self.to_lon],
            mode=self.config.interpolation_method,
            scale_factor=self.config.upscale,
        )

        # Use Transposed Convolutions to generate a target matrix.
        h = self.preprocess_model(pixel_values)
        for conv in self.cnns:
            h = conv(h)
        out_upconv = h

        if not return_dict:
            return (out_upconv + out_baseline,)

        return out_upconv + out_baseline
