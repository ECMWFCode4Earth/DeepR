import os

import diffusers
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import Repository
from tqdm import tqdm

from deepr.model.conditional_ddpm import cDDPMPipeline
from deepr.model.configs import TrainingConfig
from deepr.visualizations.plot_maps import get_figure_model_samples


def save_samples(config, model, era5: torch.Tensor, cerra: torch.Tensor, outname: str):
    """Save a set of samples."""
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
    )
    pipeline = cDDPMPipeline(unet=model, scheduler=scheduler).to(config.device)

    era5_repeated = era5.repeat(config.num_samples, 1, 1, 1)
    images = pipeline(
        images=era5_repeated,
        generator=torch.manual_seed(config.seed),
        output_type="tensor",
    ).images

    # Make a grid out of the images
    images = images.transpose(1, 3).transpose(2, 3)
    get_figure_model_samples(era5, cerra, images, filename=outname)


def train_diffusion(
    config: TrainingConfig,
    model,
    noise_scheduler: diffusers.SchedulerMixin,
    dataset: torch.utils.data.Dataset,
):
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [1 - config.validation_split, config.validation_split],
        generator=torch.Generator().manual_seed(config.seed),
    )

    # Define important objects
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, config.train_batch_size, shuffle=True, pin_memory=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, config.val_batch_size, shuffle=False, pin_memory=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = "predictia/reanalysis_downscaler"
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("Train Diffusion", config=config.__dict__)

    (
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # Get fixed samples
    val_era5, val_cerra = next(iter(val_dataloader))

    global_step = 0
    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch+1}")

        for step, (era5, cerra) in enumerate(train_dataloader):
            bs = cerra.shape[0]

            # Sample noise to add to the images
            noise = torch.randn(cerra.shape).to(config.device)

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=config.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(cerra, noise, timesteps)

            with accelerator.accumulate(model):
                model_inputs = torch.cat([noisy_images, era5], dim=1)

                # Predict the noise residual
                noise_pred = model(model_inputs, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            pred_var = noise_pred.var(keepdim=True, dim=0).mean().item()
            true_var = noise.var(keepdim=True, dim=0).mean().item()
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
                "mean_pred_noise": noise_pred.mean().item(),
                "mean_var_ratio": true_var / pred_var,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images
        if accelerator.is_main_process:
            is_last_epoch = epoch == config.num_epochs - 1

            if (epoch + 1) % config.save_image_epochs == 0 or is_last_epoch:
                test_dir = os.path.join(config.output_dir, "samples")
                os.makedirs(test_dir, exist_ok=True)
                save_samples(
                    config,
                    accelerator.unwrap_model(model),
                    val_era5,
                    val_cerra,
                    outname=f"{test_dir}/{epoch+1:04d}.png",
                )

            if (epoch + 1) % config.save_model_epochs == 0 or is_last_epoch:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch+1}", blocking=True)
                else:
                    model.save_pretrained(config.output_dir)

    accelerator.end_training()
