import logging
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

logger = logging.getLogger(__name__)


class ConvBilinearConfig(PretrainedConfig):
    model_type = "conv_bilineal"

    attribute_map = {
        "hidden_size": "embed_dim",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        num_channels: int = 1,
        num_layers: int = 3,
        hidden_kernel: int = 64,
        hidden_ch: int = 64,
        upscale: int = 1,
        image_size: Tuple[int] = None,
        conv_channels: List[int] = [],
        conv_kernel_size: List[int] = [],
        conv_stride: List[int] = None,
        conv_padding: List[str] = None,
    ):
        super().__init__()

        assert len(conv_channels) == len(
            conv_kernel_size
        ), "Unconsistent number of layers inferred."

        if conv_padding is not None:
            assert len(conv_padding) == len(
                conv_channels
            ), "Unconsistent number of layers inferred."
        else:
            conv_padding = ["same"] * len(conv_channels)

        if conv_stride is not None:
            assert len(conv_stride) == len(
                conv_channels
            ), "Unconsistent number of layers inferred."
        else:
            conv_stride = [1] * len(conv_channels)

        self.num_channels = num_channels
        self.num_layers = num_layers
        self.upscale = upscale
        self.image_size = image_size
        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.hidden_kernel = hidden_kernel
        self.hidden_ch = hidden_ch


class ConvBilinear(PreTrainedModel):
    config_class = ConvBilinearConfig
    base_model_prefix = "convswin2sr"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def __init__(self, config: ConvBilinearConfig):
        super().__init__(config)

        self.input_pixels_shape = np.array(config.image_size)

        convs: List[nn.Module] = []
        in_channels = config.num_channels
        for i in range(len(config.conv_kernel_size)):
            convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=config.conv_channels[i],
                    kernel_size=config.conv_kernel_size[i],
                    stride=config.conv_stride[i],
                    padding=config.conv_padding[i],
                )
            )
            in_channels = config.conv_channels[i]
        convs.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=config.num_channels,
                # 12 are the non-overlapping ERA5 pixels in each dimension
                kernel_size=(12 * config.upscale + 1, 12 * config.upscale + 1),
            )
        )

        self.cnns = nn.ModuleList(convs)
        self.agg_cnn = nn.Conv2d(
            in_channels=2 * config.num_channels,
            out_channels=config.num_channels,
            kernel_size=1,
            padding="same",
        )

        layers = []
        for i in range(config.num_layers):
            in_ch = config.num_channels if i == 0 else config.hidden_ch
            out_ch = (
                config.hidden_ch if i < config.num_layers - 1 else config.num_channels
            )
            layers.append(
                (
                    f"conv{i}",
                    nn.Conv2d(in_ch, out_ch, config.hidden_kernel, padding="same"),
                )
            )
            layers.append((f"relu{i}", nn.ReLU()))
        self.denoise_model = nn.Sequential(OrderedDict(layers))
        super().post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        extra_pixels = pixel_values.shape[-2:] - self.input_pixels_shape
        up_extra_pixels = self.config.upscale * extra_pixels
        lat_pixels_one_side = int(up_extra_pixels[0] // 2)
        lon_pixels_one_side = int(up_extra_pixels[1] // 2)

        up_pixels = torch.nn.functional.interpolate(
            pixel_values, mode="bicubic", scale_factor=self.config.upscale
        )

        # From upscaled image concatenate 2 channels.
        h = up_pixels

        # Channel 1: select center of image
        up_pixels_center = up_pixels[
            ...,
            lat_pixels_one_side:-lat_pixels_one_side,
            lon_pixels_one_side:-lon_pixels_one_side,
        ]

        # Channel 2: apply convolutions
        for conv in self.cnns:
            h = conv(h)

        interm = self.agg_cnn(torch.cat([up_pixels_center, h], dim=1))

        # Apply Denoising Swin2SR
        output = self.denoise_model(interm)

        if not return_dict:
            return (output,)

        return output
