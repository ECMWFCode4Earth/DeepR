import os
from typing import Type

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, logging
from huggingface_hub import Repository
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from deepr.model.configs import TrainingConfig
from deepr.model.utils import get_hour_embedding

logger = logging.get_logger(__name__, log_level="INFO")


def train_diffusion(
    config: TrainingConfig,
    model,
    noise_scheduler: diffusers.SchedulerMixin,
    dataset: torch.utils.data.IterableDataset,
    dataset_val: torch.utils.data.IterableDataset,
    obs_model: Type[torch.nn.Module] = None,
    dataset_info: dict = None,
):
    hparams = config.__dict__  # | dataset_info
    number_model_params = sum([np.prod(m.size()) for m in model.parameters()])
    if "number_model_params" not in hparams:
        hparams["number_model_params"] = int(number_model_params)

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
        cpu=config.device == "cpu",
    )

    torch.cuda.empty_cache()
    accelerator.free_memory()  # Free all lingering references

    # Define important objects
    train_dataloader = torch.utils.data.DataLoader(
        dataset, config.batch_size, pin_memory=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset_val, config.batch_size, pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo = Repository(
                config.output_dir,
                clone_from=config.hf_repo_name,
                token=os.getenv("HF_TOKEN"),
            )
            repo.git_pull()
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("Train Denoising Diffusion Model", config=hparams)

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
    val_era5, val_cerra, val_times = next(iter(val_dataloader))
    if config.batch_size > 4:
        val_era5, val_cerra, val_times = val_era5[:4], val_cerra[:4], val_times[:4]

    tf_writter = accelerator.get_tracker("tensorboard").writer
    logger.info(f"Number of parameters: {number_model_params}")
    global_step = 0
    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader) + len(val_dataloader),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch+1}")

        for era5, cerra, times in train_dataloader:
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

            # Add noise to the clean images according to the noise magnitude at each t
            noisy_images = noise_scheduler.add_noise(cerra, noise, timesteps)

            # Encode hour
            emb_size = model.down_blocks[0].resnets[0].conv1.in_channels * 4
            hour_emb = get_hour_embedding(
                times[:, :1], config.hour_embed_type, emb_size
            )
            if hour_emb is not None:
                hour_emb = hour_emb.to(config.device).squeeze()

            # Get ERA5 of the same shape as CERRA: A) trained model, B) baseline interp.
            if obs_model is not None:
                up_era5 = obs_model(era5)
            else:
                up_era5 = F.interpolate(era5, scale_factor=5, mode="bicubic")
                l_lat, l_lon = (np.array(up_era5.shape[-2:]) - cerra.shape[-2:]) // 2
                r_lat = None if l_lat == 0 else -l_lat
                r_lon = None if l_lon == 0 else -l_lon
                up_era5 = up_era5[..., l_lat:r_lat, l_lon:r_lon]

            # Predict the noise residual
            with accelerator.accumulate(model):
                model_inputs = torch.cat([noisy_images, up_era5], dim=1)

                # Predict the noise residual
                noise_pred = model(
                    model_inputs,
                    timesteps,
                    return_dict=False,
                    class_labels=hour_emb,
                )[0]
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
                "loss_vs_step": loss.detach().item(),
                "lr_vs_step": lr_scheduler.get_last_lr()[0],
                "step": global_step,
                "bias_perc_vs_step": (noise - noise_pred).mean().item()
                / noise.mean().item(),
                "mean_var_ratio_vs_step": true_var / pred_var,
                "epoch_vs_step": epoch,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # Evaluate
        loss, true_var, pred_var, bias, mean_pred = [], [], [], [], []
        for era5, cerra, times in val_dataloader:
            bs = cerra.shape[0]
            noise = torch.randn(cerra.shape).to(config.device)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=config.device,
            ).long()
            noisy_images = noise_scheduler.add_noise(cerra, noise, timesteps)

            # Validation: Encode hour
            emb_size = model.down_blocks[0].resnets[0].conv1.in_channels * 4
            hour_emb = get_hour_embedding(
                times[:, :1], config.hour_embed_type, emb_size
            )
            if hour_emb is not None:
                hour_emb = hour_emb.to(config.device).squeeze()

            # Predict the noise residual
            with torch.no_grad():
                if obs_model is not None:
                    up_era5 = obs_model(era5)
                else:
                    up_era5 = F.interpolate(era5, scale_factor=5, mode="bicubic")
                    l_lat, l_lon = (
                        np.array(up_era5.shape[-2:]) - cerra.shape[-2:]
                    ) // 2
                    r_lat = None if l_lat == 0 else -l_lat
                    r_lon = None if l_lon == 0 else -l_lon
                    up_era5 = up_era5[..., l_lat:r_lat, l_lon:r_lon]

                model_inputs = torch.cat([noisy_images, up_era5], dim=1)

                # Predict the noise residual
                noise_pred = model(
                    model_inputs,
                    timesteps,
                    return_dict=False,
                    class_labels=hour_emb,
                )[0]
                loss.append(F.mse_loss(noise_pred, noise).mean().item())

            pred_var.append(noise_pred.var(keepdim=True, dim=0).mean().item())
            true_var.append(noise.var(keepdim=True, dim=0).mean().item())
            bias.append((noise - noise_pred).mean().item())
            mean_pred.append(noise_pred.mean().item())
            progress_bar.update(1)

        logs = {
            "val_loss_vs_epoch": sum(loss) / len(loss),
            "val_bias_perc_vs_epoch": sum(bias) / sum(mean_pred),
            "val_mean_var_ratio_vs_epoch": sum(true_var) / sum(pred_var),
            "epoch": epoch,
        }
        accelerator.log(logs, step=epoch)
        progress_bar.close()

        # After each epoch you optionally sample some demo images
        if accelerator.is_main_process:
            is_last_epoch = epoch == config.num_epochs - 1

            if epoch < 0:  # Never
                tf_writter.add_graph(
                    accelerator.unwrap_model(model), (model_inputs, timesteps)
                )

            if (epoch + 1) % config.save_model_epochs == 0 or is_last_epoch:
                model.save_pretrained(config.output_dir)
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch+1}", blocking=True)

    accelerator.end_training()
    return model, config.hf_repo_name
