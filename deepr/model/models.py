from typing import Tuple

import diffusers
from torch import nn

from deepr.utilities.logger import get_logger

logger = get_logger(__name__)


def load_trained_model(class_name: str = None, model_dir: str = None) -> nn.Module:
    """Load a trained model and return it in evaluation mode.

    Args:
    ----
        class_name (str): Name of the model class. Options are
        model_dir (str): Directory where the model is stored.

    Returns:
    -------
        nn.Module: the model in evaluation mode.
    """
    if class_name is None or model_dir is None:
        return None
    elif class_name.lower() == "convbaseline":
        from deepr.model.conv_baseline import ConvBaseline

        model = ConvBaseline.from_pretrained(model_dir)
    elif class_name.lower() == "convswin2sr":
        from deepr.model.conv_swin2sr import ConvSwin2SR

        model = ConvSwin2SR.from_pretrained(model_dir)
    else:
        logger.warning(
            f"The class_name {class_name} is not implemented. Options are 'convbaseline' and 'convswin2sr'."
        )
        return None
    model.eval()
    return model


def get_neural_network(
    class_name: str,
    kwargs: dict,
    input_shape: Tuple[int] = None,
    sample_size: Tuple[int] = None,
    out_channels: int = None,
) -> nn.Module:
    """Get neural network.

    Given a class name and a dictionary of keyword arguments, returns an instance of a
    neural network. Current options are: "UNet".

    Arguments
    ---------
    class_name : str
        The name of the neural network class to use.
    kwargs : dict
        Dictionary of keyword arguments to pass to the neural network constructor.
    input_shape : Optional[tuple]
        Sample size of the input samples.
    sample_size : Optional[tuple]
        Sample size of the target samples.
    out_channels : Optional[int]
        Output channels of the target samples.

    Returns
    -------
    model: nn.Module
        An instance of a neural network.

    Raises:
    ------
        NotImplementedError: If the specified neural network class is not implemented.
    """
    if "sample_size" in kwargs:
        kwargs["sample_size"] = tuple(kwargs["sample_size"])
    elif sample_size is None:
        raise ValueError(f"sample_size must be specified for {class_name}")
    else:
        kwargs["sample_size"] = sample_size

    if "out_channels" not in kwargs and out_channels is not None:
        kwargs["out_channels"] = out_channels

    if class_name.lower() == "unet":
        from deepr.model.unet import UNet

        return UNet(**kwargs)
    elif class_name.lower() == "convswin2sr":
        from deepr.model.conv_swin2sr import ConvSwin2SR, ConvSwin2SRConfig

        kwargs["num_channels"] = kwargs.pop("out_channels")
        if input_shape is not None:
            kwargs["input_shape"] = input_shape

        cfg = ConvSwin2SRConfig(**kwargs)
        return ConvSwin2SR(cfg)
    elif class_name.lower() == "convbaseline":
        from deepr.model.conv_baseline import ConvBaseline, ConvBaselineConfig

        kwargs["num_channels"] = kwargs.pop("out_channels")
        if input_shape is not None:
            kwargs["input_shape"] = input_shape

        cfg = ConvBaselineConfig(**kwargs)
        return ConvBaseline(cfg)
    elif class_name.split(".")[0].lower() == "diffusers":
        import diffusers

        return diffusers.__dict__[class_name.split(".")[1]](**kwargs)
    elif class_name.split(".")[0].lower() == "transformers":
        import transformers

        return transformers.__dict__[class_name.split(".")[1]](**kwargs)
    else:
        raise NotImplementedError(f"{class_name} is not implemented")


def get_hf_scheduler(class_name: str, kwargs: dict) -> diffusers.SchedulerMixin:
    logger.info(f"Loading scheduler {class_name}.")
    return getattr(diffusers, class_name)(**kwargs)
