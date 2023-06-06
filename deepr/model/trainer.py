import os

import diffusers
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import Repository
from PIL import Image
from tqdm import tqdm

from deepr.model.configs import TrainingConfig


def save_samples(config, model, outname: str):
    """Save a set of samples."""
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
    )
    pipeline = DDPMPipeline(unet=model, scheduler=scheduler).to(config.device)

    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    w, h = images[0].size
    cols, rows = 4, config.eval_batch_size // 4
    image_grid = Image.new("L", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        image_grid.paste(image, box=(i % cols * w, i // cols * h))

    # Save the image
    image_grid.save(outname)


def train_diffusion(
    config: TrainingConfig,
    model,
    noise_scheduler: diffusers.SchedulerMixin,
    dataset: torch.utils.data.Dataset,
):
    # Define important objects
    train_dataloader = torch.utils.data.DataLoader(
        dataset, config.train_batch_size, shuffle=True, pin_memory=True
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

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch+1}")

        for step, (_, cerra) in enumerate(train_dataloader):
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

            # Add noise to the clean images according to the noise magnitude at each
            # timestep (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(cerra, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
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
                    outname=f"{test_dir}/{epoch+1:04d}.png",
                )

            if (epoch + 1) % config.save_model_epochs == 0 or is_last_epoch:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch+1}", blocking=True)
                else:
                    model.save_pretrained(config.output_dir)
