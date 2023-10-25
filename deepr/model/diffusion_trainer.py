import math
import os
from typing import Type

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import Repository
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from deepr.model.configs import TrainingConfig
from deepr.model.utils import get_hour_embedding
from deepr.utilities.logger import get_logger
from deepr.validation.sample_predictions import diffusion_callback

logger = get_logger(__name__)

load_dotenv(find_dotenv())


def train_diffusion(
    config: TrainingConfig,
    model,
    noise_scheduler: diffusers.SchedulerMixin,
    dataset: torch.utils.data.IterableDataset,
    dataset_val: torch.utils.data.IterableDataset,
    obs_model: Type[torch.nn.Module] = None,
    dataset_info: dict = None,
    input_scaler=None,
    label_scaler=None,
):
    hparams = config.__dict__ | dataset_info
    hparams.pop("__pydantic_initialised__", None)
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
        dataset, config.batch_size, pin_memory=True, num_workers=config.num_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset_val, config.batch_size, pin_memory=True, num_workers=config.num_workers
    )
    logger.info("DataLoaders created successfully!")

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

    noise_scheduler.save_pretrained(config.output_dir)

    (
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    if obs_model is not None:
        obs_model = accelerator.prepare(obs_model)
        obs_model.eval()

    accelerator.get_tracker("tensorboard").writer
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

            # Encode hour
            hour_emb = get_hour_embedding(
                times[:, :1], config.hour_embed_type, config.hour_embed_size
            )
            if hour_emb is not None:
                hour_emb = hour_emb.to(config.device).squeeze()

            # Get ERA5 of the same shape as CERRA: A) trained model, B) baseline interp.
            if obs_model is not None:
                with torch.no_grad():
                    era5 = obs_model(era5)[0]
            else:
                era5 = F.interpolate(era5, scale_factor=5, mode="bicubic")
                l_lat, l_lon = (np.array(era5.shape[-2:]) - cerra.shape[-2:]) // 2
                r_lat = None if l_lat == 0 else -l_lat
                r_lon = None if l_lon == 0 else -l_lon
                era5 = era5[..., l_lat:r_lat, l_lon:r_lon]

            if config.instance_norm:
                m = era5.mean((1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
                s = era5.std((1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
                era5 = (era5 - m) / s
                cerra = (cerra - m) / s

            if config.learn_residuals:
                cerra = cerra - era5

            # Add noise to the clean images according to the noise magnitude at each t
            noisy_images = noise_scheduler.add_noise(cerra, noise, timesteps)

            # Predict the noise residual
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(
                    torch.cat([noisy_images, era5], dim=1),
                    timesteps,
                    return_dict=False,
                    class_labels=hour_emb,
                )[0]
                loss = F.mse_loss(noise_pred, noise)

                if torch.isfinite(loss):
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

            if math.isnan(logs["loss_vs_step"]):
                tensors_to_assess = {
                    "low-res image": era5,
                    "target noise": noise,
                    "noisy sample": noisy_images,
                    "prediction": noise_pred,
                }
                for name, t in tensors_to_assess.items():
                    if t.isnan().any():
                        nans = t.isnan()
                        while nans.ndim > 1:
                            nans = nans.any(dim=-1)
                        nan_sample = nans.nonzero(as_tuple=True)[0].detach().tolist()
                        nan_times = times[nan_sample].detach().tolist()
                        nan_timesteps = timesteps[nan_sample].detach().tolist()
                        raise ValueError(
                            f"The training loss has collapsed to NaN values. The "
                            f"{nan_sample}-th {name} of the batch has NaN's, which"
                            f" corresponds to time={nan_times}, and timestep="
                            f"{nan_timesteps}:\n{t}"
                        )
                    elif t.isinf().any():
                        infs = t.isinf()
                        while infs.ndim > 1:
                            infs = infs.any(dim=-1)
                        inf_sample = infs.nonzero(as_tuple=True)[0].detach().tolist()
                        inf_times = times[inf_sample].detach().tolist()
                        inf_timesteps = timesteps[inf_sample].detach().tolist()
                        raise ValueError(
                            f"The training loss has collapsed to NaN values. The "
                            f"{inf_sample}-th {name} of the batch has Inf's, which"
                            f" corresponds to time={inf_times}, and timestep="
                            f"{inf_timesteps}:\n{t}"
                        )

                raise ValueError("The training loss has collapsed to NaN values.")

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
                    era5 = obs_model(era5)[0]
                else:
                    era5 = F.interpolate(era5, scale_factor=5, mode="bicubic")
                    l_lat, l_lon = (np.array(era5.shape[-2:]) - cerra.shape[-2:]) // 2
                    r_lat = None if l_lat == 0 else -l_lat
                    r_lon = None if l_lon == 0 else -l_lon
                    era5 = era5[..., l_lat:r_lat, l_lon:r_lon]

                # Instance normalization
                if config.instance_norm:
                    m = era5.mean((1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
                    s = era5.std((1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
                    era5 = (era5 - m) / s
                    cerra = (cerra - m) / s

                if config.learn_residuals:
                    cerra = cerra - era5

                noisy_images = noise_scheduler.add_noise(cerra, noise, timesteps)

                # Predict the noise residual
                noise_pred = model(
                    torch.cat([noisy_images, era5], dim=1),
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
            epoch == config.num_epochs - 1

            if config.is_save_images_time(epoch):
                try:
                    era5, cerra, times = next(iter(val_dataloader))
                    if config.batch_size > 1:
                        era5, cerra, times = era5[:1], cerra[:1], times[:1]
                    diffusion_callback(
                        model,
                        noise_scheduler,
                        era5,
                        cerra,
                        times,
                        inference_steps=1000,
                        freq_timesteps_frame=4,
                        scaler_func=label_scaler.apply_inverse_scaler,
                        output_dir=config.output_dir,
                        epoch=epoch + 1,
                        obs_model=obs_model,
                        learn_residuals=config.learn_residuals,
                        instance_norm=config.instance_norm,
                        hour_embed_type=config.hour_embed_type,
                        hour_embed_dim=config.hour_embed_dim,
                    )
                    del era5, cerra, times
                except Exception as e:
                    logger.error(f"Error when saving images at epoch {epoch}: \n{e}")

            if config.is_save_model_time(epoch):
                model.save_pretrained(config.output_dir)
                if config.push_to_hub:
                    try:
                        repo.push_to_hub(
                            commit_message=f"Epoch {epoch+1}", blocking=True
                        )
                    except Exception as e:
                        logger.error(f"Error when pushing model to hub: \n{e}")

    accelerator.end_training()
    return model, config.hf_repo_name
