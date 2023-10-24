from inspect import signature
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

from deepr.model.utils import get_hour_embedding


class cDDPMPipeline(DiffusionPipeline):
    r"""
    DDPM conditioned on images.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation
    for the generic methods the library implements for all the pipelines (such as
    downloading or saving, running on a particular device, etc.).

    Parameters
    ----------
    unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
    scheduler ([`SchedulerMixin`]):
        A scheduler to be used in combination with `unet` to denoise the encoded
        image. Can be one of [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(
        self,
        unet,
        scheduler,
        obs_model=None,
        baseline_interpolation_method: Optional[str] = "bicubic",
        learn_residuals: Optional[bool] = False,
        hour_embed_type: [Optional] = "class",
        hour_embed_dim: Optional[int] = 64,
        instance_norm: Optional[bool] = False,
    ):
        super().__init__()
        self.baseline_interpolation_method = baseline_interpolation_method
        self.hour_embed_type = hour_embed_type
        self.hour_embed_dim = hour_embed_dim
        self.instance_norm = instance_norm
        self.learn_residuals = learn_residuals
        self.register_modules(unet=unet, scheduler=scheduler, obs_model=obs_model)

    @torch.no_grad()
    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        eta: Optional[float] = 1.0,
        class_labels: Optional[List[int]] = None,
        output_type: Optional[str] = "pil",
        saving_freq_interm: int = 0,
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        # Get batch size
        if isinstance(images, torch.Tensor):
            batch_size = images.shape[0]
        elif isinstance(images, list):
            batch_size = len(images)
        else:
            raise ValueError(f"Unsupported type {type(images)} for `images` argument.")

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.out_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (
                batch_size,
                self.unet.config.out_channels,
                *self.unet.config.sample_size,
            )

        if self.obs_model is not None:
            self.obs_model = self.obs_model.to(self.device)
            up_images = self.obs_model(images.to(self.device))[0].to(self.device)
        else:
            up_images = F.interpolate(
                images, scale_factor=5, mode=self.baseline_interpolation_method
            )
            l_lat, l_lon = (np.array(up_images.shape[-2:]) - image_shape[-2:]) // 2
            r_lat = None if l_lat == 0 else -l_lat
            r_lon = None if l_lon == 0 else -l_lon
            up_images = up_images[..., l_lat:r_lat, l_lon:r_lon].to(self.device)

        if self.instance_norm:
            m = up_images.mean((1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
            s = up_images.std((1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
            up_images = (up_images - m) / s

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            latents = randn_tensor(image_shape, generator=generator)
            latents = latents.to(self.device)
        else:
            latents = randn_tensor(image_shape, generator=generator, device=self.device)

        # support for DDIM scheduler
        accepts_eta = "eta" in set(signature(self.scheduler.step).parameters.keys())
        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        # Support for LSMDiscreteScheduler
        if "generator" in set(signature(self.scheduler.step).parameters.keys()):
            extra_kwargs["generator"] = generator

        # Hour encoding. Passed to NN as class labels
        if class_labels is not None:
            class_labels = get_hour_embedding(
                class_labels, self.hour_embed_type, self.hour_embed_dim
            )
            class_labels = class_labels.to(self.device).squeeze()

        # set step values
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = latents * self.scheduler.init_noise_sigma

        intermediate_images = []
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            if saving_freq_interm > 0 and i % saving_freq_interm == 0:
                intermediate_images.append(latents.cpu())

            latents_input = torch.cat([latents, up_images], axis=1)
            latents_input = self.scheduler.scale_model_input(latents_input, t)

            # 1. predict noise model_output
            model_output = self.unet(latents_input, t, class_labels=class_labels).sample

            # 2. compute previous image: x_t -> x_t-1
            latents = self.scheduler.step(
                model_output, t, latents, **extra_kwargs
            ).prev_sample

        if saving_freq_interm > 0:
            intermediate_images.append(latents.cpu())
            intermediate_images = torch.cat(intermediate_images, dim=1)

        if self.learn_residuals:
            latents = latents + up_images

        if self.instance_norm:
            latents = latents * s + m

        image = latents.cpu().numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        elif output_type == "tensor":
            image = torch.tensor(image)

        if not return_dict:
            return image, intermediate_images if saving_freq_interm > 0 else (image,)

        return ImagePipelineOutput(images=image)
