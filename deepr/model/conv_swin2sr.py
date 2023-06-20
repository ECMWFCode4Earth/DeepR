import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel, Swin2SRConfig, Swin2SRForImageSuperResolution

logger = logging.getLogger(__name__)


class ConvSwin2SRConfig(Swin2SRConfig):
    model_type = "conv_swin2sr"

    attribute_map = {
        "hidden_size": "embed_dim",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 1,
        num_channels: int = None,
        embed_dim: int = 180,
        depths: List[int] = [6, 6, 6, 6, 6, 6],
        num_heads: List[int] = [6, 6, 6, 6, 6, 6],
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        drop_path_rate: float = 0.1,
        hidden_act: str = "gelu",
        use_absolute_embeddings: bool = False,
        initializer_range: bool = 0.02,
        layer_norm_eps: float = 1e-5,
        upscale: int = 2,
        img_range: float = 1.0,
        resi_connection: str = "1conv",
        upsampler: str = "pixelshuffle",
        conv_channels: List[int] = [],
        conv_kernel_size: List[int] = [],
        conv_stride: List[int] = None,
        conv_padding: List[str] = None,
        **kwargs,
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            drop_path_rate=drop_path_rate,
            hidden_act=hidden_act,
            use_absolute_embeddings=use_absolute_embeddings,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            upscale=upscale,
            img_range=img_range,
            resi_connection=resi_connection,
            upsampler=upsampler,
            **kwargs,
        )

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

        self.conv_channels = conv_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding

    def swin2sr_kwargs(self):
        logger.info(
            f"The Swin2SR(x{self.upscale}) model should receive pixel values of shape"
            f"{self.image_size}."
        )
        return Swin2SRConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            embed_dim=self.embed_dim,
            depths=self.depths,
            num_heads=self.num_heads,
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            drop_path_rate=self.drop_path_rate,
            hidden_act=self.hidden_act,
            use_absolute_embeddings=self.use_absolute_embeddings,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            upscale=self.upscale,
            img_range=self.img_range,
            resi_connection=self.resi_connection,
            upsampler=self.upsampler,
        )


class ConvSwin2SR(PreTrainedModel):
    config_class = ConvSwin2SRConfig
    base_model_prefix = "convswin2sr"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _scale_factor = 5

    def __init__(self, config: ConvSwin2SRConfig):
        super().__init__(config)
        self.config = config

        self.input_pixels_shape = np.array(self.config.image_size) / self._scale_factor

        in_channels = config.num_channels
        convs: List[nn.Module] = []
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
                out_channels=self.config.num_channels,
                # 12 are the non-overlapping ERA5 pixels in each dimension
                kernel_size=(12 * self._scale_factor + 1, 12 * self._scale_factor + 1),
            )
        )

        self.cnns = nn.ModuleList(convs)
        self.agg_cnn = nn.Conv2d(
            in_channels=2 * self.config.num_channels,
            out_channels=self.config.num_channels,
            kernel_size=1,
            padding="same",
        )

        self.swin = Swin2SRForImageSuperResolution(config.swin2sr_kwargs())
        super().post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        extra_pixels = pixel_values.shape[-2:] - self.input_pixels_shape
        up_extra_pixels = self._scale_factor * extra_pixels
        lat_pixels_one_side = int(up_extra_pixels[0] // 2)
        lon_pixels_one_side = int(up_extra_pixels[1] // 2)

        up_pixels = torch.nn.functional.interpolate(
            pixel_values, mode="bicubic", scale_factor=self._scale_factor
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

        intermediate = self.agg_cnn(torch.cat([up_pixels_center, h], dim=1))

        # Apply Denoising Swin2SR
        (out,) = self.swin(
            pixel_values=intermediate,
            head_mask=head_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return (out + up_pixels_center,)
