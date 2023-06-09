import os
from typing import Dict

import evaluate
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import Repository
from tqdm import tqdm

from deepr.model.configs import TrainingConfig
from deepr.visualizations.plot_maps import get_figure_model_samples

repo_name = "predictia/europe_reanalysis_downscaler_nn"


def save_samples(
    model,
    era5: torch.Tensor,
    cerra: torch.Tensor,
    times: torch.Tensor,
    outname: str,
):
    """Save a set of samples."""
    with torch.no_grad():
        images = model(era5, return_dict=False)[0]

    # Make a grid out of the images
    sample_names = [f"{t[0]:d}H {t[1]:02d}-{t[2]:02d}-{t[3]:04d}" for t in times]
    return get_figure_model_samples(
        era5.cpu(),
        cerra.cpu(),
        images.cpu(),
        column_names=sample_names,
        filename=outname,
    )


def train_nn(
    config: TrainingConfig,
    model,
    train_dataset: torch.utils.data.DataLoader,
    test_dataset: torch.utils.data.DataLoader,
    dataset_info: Dict = None,
):
    # Define important objects
    dataloader = torch.utils.data.DataLoader(
        train_dataset, config.train_batch_size, pin_memory=True
    )
    dataloader_val = torch.utils.data.DataLoader(
        test_dataset, config.val_batch_size, pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(dataloader) * config.num_epochs),
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        if config.push_to_hub:
            repo = Repository(
                config.output_dir, clone_from=repo_name, token=os.getenv("HF_TOKEN")
            )
            repo.git_pull()
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("Train Neural Network", config=config.__dict__)

    (
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(model, optimizer, dataloader, dataloader_val, lr_scheduler)

    # Get fixed samples
    val_era5, val_cerra, val_times = next(iter(val_dataloader))

    tf_writter = accelerator.get_tracker("tensorboard").writer
    global_step = 0
    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader), disable=not accelerator.is_local_main_process
        )
        progress_bar.set_description(f"Epoch {epoch+1}")

        for era5, cerra, times in train_dataloader:
            # Predict the noise residual
            with accelerator.accumulate(model):
                cerra_pred = model(era5, return_dict=False)[0]
                loss = F.l1_loss(cerra_pred, cerra)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            pred_var = cerra_pred.var(keepdim=True, dim=0).mean().item()
            true_var = cerra.var(keepdim=True, dim=0).mean().item()
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
                "bias_perc": (cerra - cerra_pred).mean().item() / cerra.mean().item(),
                "mean_var_ratio": true_var / pred_var,
                "epoch": epoch,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            tf_writter.add_histogram("cerra prediction", cerra_pred, global_step)
            tf_writter.add_histogram("cerra", cerra, global_step)
            global_step += 1

        progress_bar.close()

        # After each epoch you optionally sample some demo images
        if accelerator.is_main_process:
            is_last_epoch = epoch == config.num_epochs - 1

            if (epoch + 1) % config.save_image_epochs == 0 or is_last_epoch:
                test_dir = os.path.join(config.output_dir, "samples")
                os.makedirs(test_dir, exist_ok=True)
                fig = save_samples(
                    accelerator.unwrap_model(model),
                    val_era5,
                    val_cerra,
                    val_times,
                    outname=f"{test_dir}/nn_{epoch+1:04d}.png",
                )
                tf_writter.add_figure("Predictions", fig, global_step=epoch)

            if (epoch + 1) % config.save_model_epochs == 0 or is_last_epoch:
                if config.push_to_hub:
                    repo.push_to_hub(
                        repo_id=repo_name,
                        commit_message=f"Epoch {epoch+1}",
                        blocking=True,
                    )
                else:
                    model.save_pretrained(config.output_dir)

    accelerator.end_training()

    mse = evaluate.load("mse", "multilist")
    for era5, cerra, times in val_dataloader:
        # Predict the noise residual
        with torch.no_grad():
            pred = model(era5, times[:, 0], return_dict=False)[0]
            mse.add_batches(
                references=cerra.reshape((cerra.shape[0], -1)),
                predictions=pred.reshape((pred.shape[0], -1)),
            )

    val_mse = mse.compute()
    hparams = config.__dict__ + dataset_info if dataset_info is not None else {}
    tf_writter.add_hparams(hparams, {"test_mse": val_mse["mse"]})
    if accelerator.is_main_process:
        if config.push_to_hub:
            evaluate.save(
                config.output_dir,
                experiment="Train Neural Network",
                **val_mse,
                **hparams,
            )
            evaluate.push_to_hub(
                model_id=repo_name,
                metric_type="mse",
                metric_name="MSE",
                metric_value=val_mse["mse"],
                dataset_split="test",
                task_type="super-resolution",
                task_name="Super Resolution",
            )
