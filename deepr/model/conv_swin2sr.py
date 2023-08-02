import logging
from math import ceil, log2
from typing import List, Optional

import numpy as np
import torch
from torch import nn
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
        interpolation_method: str = "bicubic",
        num_high_res_covars: int = 0,
        **kwargs,
    ):
        self.interpolation_method = interpolation_method
        self.num_high_res_covars = num_high_res_covars

        if "real_upscale" in kwargs.keys():
            self.real_upscale = kwargs["real_upscale"]
        else:
            self.real_upscale = upscale

        if "sample_size" in kwargs.keys():
            self.image_size = tuple(
                map(lambda x: x // self.real_upscale, kwargs["sample_size"])
            )
        else:
            self.image_size = image_size

        upscale_power2 = int(ceil(log2(self.real_upscale)))
        super().__init__(
            image_size=self.image_size,
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
            upscale=2**upscale_power2,
            img_range=img_range,
            resi_connection=resi_connection,
            upsampler=upsampler,
            **kwargs,
        )

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

    def __init__(self, config: ConvSwin2SRConfig):
        super().__init__(config)
        self.config = config

        # Define center region to upscale
        extra_pixels = np.array(config.input_shape) - config.image_size
        self.from_lat = int(extra_pixels[0] // 2)
        self.from_lon = int(extra_pixels[1] // 2)
        self.to_lat = -self.from_lat if self.from_lat > 0 else None
        self.to_lon = -self.from_lon if self.from_lon > 0 else None

        # Set preprocess layer to match the output shapes
        self.input_upconv_shape = np.array(config.sample_size) / config.upscale
        kernel_size = (
            np.array(config.input_shape) - self.input_upconv_shape + 1
        ).astype(int)
        self.preprocess_model = nn.Conv2d(
            config.num_channels,
            config.num_channels,
            kernel_size=tuple(kernel_size),
            padding=0,
        )

        self.swin = Swin2SRForImageSuperResolution(config.swin2sr_kwargs())

        if self.config.num_high_res_covars > 0:
            self.merge_covars_interp = nn.Conv2d(
                config.num_channels + config.num_high_res_covars, config.num_channels, 1
            )
            self.merge_covars_swin2sr = nn.Conv2d(
                config.num_channels + config.num_high_res_covars, config.num_channels, 1
            )

        super().post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        covariables: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        out_baseline = torch.nn.functional.interpolate(
            pixel_values[..., self.from_lat : self.to_lat, self.from_lon : self.to_lon],
            mode=self.config.interpolation_method,
            scale_factor=self.config.real_upscale,
        )

        if self.config.num_high_res_covars > 0 and covariables is not None:
            covariables = torch.tile(covariables, (out_baseline.shape[0], 1, 1, 1))
            out_baseline = self.merge_covars_interp(
                torch.cat([out_baseline, covariables], dim=1)
            )

        h = self.preprocess_model(pixel_values)

        # Apply Denoising Swin2SR
        (out_swin2sr,) = self.swin(
            pixel_values=h,
            head_mask=head_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        if self.config.num_high_res_covars > 0 and covariables is not None:
            out_swin2sr = self.merge_covars_swin2sr(
                torch.cat([out_swin2sr, covariables], dim=1)
            )

        if not return_dict:
            return (out_baseline + out_swin2sr,)

        return out_baseline + out_swin2sr
