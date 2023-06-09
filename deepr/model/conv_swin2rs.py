import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution

logger = logging.getLogger(__name__)


class ConvSwin2RS(nn.Module):
    def __init__(
        self,
        conv_channels: List[int],
        conv_kernel_size: List[int],
        conv_stride: List[int],
        conv_padding: List[str],
        swin2rs_kawrgs: Dict[str, Any],
        out_channels: int,
        sample_size: Optional[Union[int, Tuple[int]]] = None,
    ):
        assert (
            len(conv_channels)
            == len(conv_stride)
            == len(conv_kernel_size)
            == len(conv_padding)
        ), "Unconsistent number of layers inferred."
        if "num_channels" not in swin2rs_kawrgs:
            swin2rs_kawrgs["num_channels"] = out_channels

        in_channels = swin2rs_kawrgs["num_channels"]
        n_layers = len(conv_kernel_size)
        assert (
            conv_channels[-1] == in_channels
        ), "Ouput channels of CNN must match the input channels of the Swin2SR."

        convs: List[nn.Module] = []
        for i in range(n_layers):
            convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=conv_channels[i],
                    kernel_size=conv_kernel_size[i],
                    stride=conv_stride[i],
                    padding=conv_padding[i],
                )
            )
            in_channels = conv_channels[i]

        super().__init__()
        self.cnn = nn.ModuleList(convs)

        swin2rs_kawrgs["image_size"] = tuple(
            (torch.tensor(sample_size) // swin2rs_kawrgs["upscale"]).tolist()
        )
        logger.info(
            f"The Swin2SR model should receive pixel values of shape "
            f"{swin2rs_kawrgs['image_size']}"
        )

        swin_config = Swin2SRConfig(**swin2rs_kawrgs)
        self.swin = Swin2SRForImageSuperResolution(swin_config)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        h = pixel_values
        for conv in self.cnn:
            h = conv(h)
        return self.swin(
            pixel_values=h,
            head_mask=head_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
