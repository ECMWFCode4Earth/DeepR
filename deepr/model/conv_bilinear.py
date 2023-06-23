import logging
from math import ceil, log2
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

logger = logging.getLogger(__name__)


class ConvBilinearConfig(PretrainedConfig):
    model_type = "convbilinear"

    attribute_map = {"hidden_size": "embed_dim"}

    def __init__(
        self,
        num_channels: int = 1,
        upscale: int = 1,
        interpolation_method: str = "bicubic",
        image_size: Tuple[int] = None,
        upblock_channels: List[int] = [64, 32],
        upblock_kernel_size: List[int] = [5, 3],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.upscale = upscale
        self.image_size = image_size
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


class ConvBilinear(PreTrainedModel):
    config_class = ConvBilinearConfig
    base_model_prefix = "convbilinear"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    upscale_ratio_upconv = 2

    def __init__(self, config: ConvBilinearConfig):
        super().__init__(config)

        self.input_pixels_shape = np.array(config.image_size)
        self.output_pixels_shape = self.input_pixels_shape * self.config.upscale
        self.input_upconv_shape = self.output_pixels_shape / (
            self.upscale_ratio_upconv**self.config.upscale_power2
        )
        kernel_size = (np.array([44, 60]) - self.input_upconv_shape + 1).astype(int)

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
        extra_pixels = pixel_values.shape[-2:] - np.array(self.config.image_size)
        lat_pixels_one_side = int(extra_pixels[0] // 2)
        lon_pixels_one_side = int(extra_pixels[1] // 2)
        to_lat = -lat_pixels_one_side if lat_pixels_one_side > 0 else None
        to_lon = -lon_pixels_one_side if lon_pixels_one_side > 0 else None

        out_baseline = torch.nn.functional.interpolate(
            pixel_values[..., lat_pixels_one_side:to_lat, lon_pixels_one_side:to_lon],
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
