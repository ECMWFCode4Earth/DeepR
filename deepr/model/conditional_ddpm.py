from inspect import signature
from typing import List, Optional, Tuple, Union

import torch
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils import randn_tensor


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

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor]],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        eta: Optional[float] = 0.0,
        output_type: Optional[str] = "pil",
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

        # set step values
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = latents * self.scheduler.init_noise_sigma

        for t in self.progress_bar(self.scheduler.timesteps):
            latents_input = torch.cat([latents, images], axis=1)
            latents_input = self.scheduler.scale_model_input(latents_input, t)

            # 1. predict noise model_output
            model_output = self.unet(latents_input, t).sample

            # 2. compute previous image: x_t -> x_t-1
            latents = self.scheduler.step(
                model_output, t, latents, generator=generator, **extra_kwargs
            ).prev_sample

        image = (latents / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        elif output_type == "tensor":
            image = torch.tensor(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
