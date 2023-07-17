import os
from typing import Dict

import matplotlib.pyplot
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, find_executable_batch_size
from accelerate.utils import LoggerType
from diffusers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import Repository
from tqdm import tqdm

from deepr.data.generator import DataGenerator
from deepr.model.configs import TrainingConfig
from deepr.utilities.logger import get_logger
from deepr.visualizations.plot_maps import get_figure_model_samples

repo_name = "predictia/cerra_tas_vqvae"

logger = get_logger(__name__)


def save_samples(
    model,
    cerra: torch.Tensor,
    output_name: str,
) -> matplotlib.pyplot.Figure:
    """
    Save a set of samples.

    Parameters
    ----------
    model : nn.Module
        The model used for generating samples.
    cerra : torch.Tensor
        The CERRA data tensor.
    output_name : str
        The output file name.

    Returns
    -------
        Figure: The figure.
    """
    with torch.no_grad():
        cerra_pred = model(cerra, return_dict=False)[0]

    figsize = 3 + 4.5 * cerra.shape[0], 8
    return get_figure_model_samples(
        cerra.cpu(), cerra_pred.cpu(), filename=output_name, fig_size=figsize
    )


def train_autoencoder(
    config: TrainingConfig,
    model,
    train_dataset: DataGenerator,
    val_dataset: DataGenerator,
    dataset_info: Dict = {},
):
    """
    Train a neural network model.

    Parameters
    ----------
    config : TrainingConfig
        The training configuration.
    model : nn.Module
        The neural network model.
    train_dataset : DataGenerator
        The training dataset.
    val_dataset : DataGenerator
        The validation dataset.
    dataset_info : Dict, optional
        Additional dataset information, by default {}.

    Returns
    -------
    model : nn.Module
        The trained model.
    repo_name : str
        The repository name.

    Notes
    -----
    This function performs the training of a neural network model using the provided
    datasets and configuration.
    """
    hparams = config.__dict__
    number_model_params = int(sum([np.prod(m.size()) for m in model.parameters()]))
    if "number_model_params" not in hparams:
        hparams["number_model_params"] = number_model_params

    model_name = model.__class__.__name__
    run_name = "Train VQ-VAE NN"

    # aim_tracker = AimTracker(run_name, logging_dir="aim://10.9.64.88:31441")
    accelerator = Accelerator(
        cpu=config.device == "cpu",
        device_placement=True,
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=[LoggerType.TENSORBOARD],  # aim_tracker
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    @find_executable_batch_size(starting_batch_size=64)
    def inner_training_loop(batch_size: int, model):
        nonlocal accelerator  # Ensure they can be used in our context
        accelerator.free_memory()  # Free all lingering references
        torch.cuda.empty_cache()

        # Define important objects
        dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size, pin_memory=True
        )
        dataloader_val = torch.utils.data.DataLoader(
            val_dataset, batch_size, pin_memory=True
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(len(dataloader) * config.num_epochs),
        )

        if accelerator.is_main_process:
            if config.push_to_hub:
                repo = Repository(
                    config.output_dir,
                    clone_from=repo_name.format(model=model_name.lower()),
                    token=os.getenv("HF_TOKEN"),
                )
                repo.git_pull()
            elif config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True)
            accelerator.init_trackers(run_name, config=hparams)
            tfboard_tracker = accelerator.get_tracker("tensorboard")

        (
            model,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            model, optimizer, dataloader, dataloader_val, lr_scheduler
        )

        # Get fixed samples
        (val_cerra,) = next(iter(val_dataloader))
        if batch_size > 4:
            val_cerra = val_cerra[:4]

        logger.info(f"Number of parameters: {number_model_params}")
        global_step = 0
        # Now you train the model
        for epoch in range(config.num_epochs):
            progress_bar = tqdm(
                total=len(train_dataloader) + len(val_dataloader),
                disable=not accelerator.is_local_main_process,
            )
            progress_bar.set_description(f"Epoch {epoch+1}")

            for (cerra,) in train_dataloader:
                # Predict the noise residual
                with accelerator.accumulate(model):
                    # Encode, quantize and decode
                    h = model.encode(cerra).latents
                    q, emb_loss, _ = model.quantize(h)
                    q = model.post_quant_conv(q)
                    cerra_pred = model.decoder(q)

                    # Calculate the loss
                    rec_loss = F.mse_loss(cerra, cerra_pred)
                    loss = emb_loss + rec_loss

                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                lo = loss.detach().item()
                logs = {
                    "loss_vs_step": lo,
                    "loss_emb_vs_step": emb_loss.detach().item(),
                    "loss_recon_vs_step": rec_loss.detach().item(),
                    "lr_vs_step": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                    "epoch": epoch,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                # tfboard_tracker.writer.add_histogram(
                #    "cerra prediction", cerra_pred, global_step
                # )
                # tfboard_tracker.writer.add_histogram("cerra", cerra, global_step)
                global_step += 1

            # Evaluate
            loss, loss_emb, loss_recs = [], [], []
            for (cerra,) in val_dataloader:
                # Predict the noise residual
                with torch.no_grad():
                    # Encode, quantize and decode
                    h = model.encode(cerra).latents
                    quant, emb_loss, _ = model.quantize(h)
                    quant2 = model.post_quant_conv(quant)
                    cerra_pred = model.decoder(quant2)

                    rec_loss = F.mse_loss(cerra, cerra_pred)

                    loss.append(emb_loss + rec_loss)
                    loss_emb.append(emb_loss)
                    loss_recs.append(rec_loss)

                progress_bar.update(1)
            torch.cuda.empty_cache()

            logs = {
                "val_loss_vs_epoch": sum(loss) / len(loss),
                "val_loss_emb_vs_epoch": sum(loss_emb) / len(loss_emb),
                "val_loss_recon_vs_epoch": sum(loss_recs) / len(loss_recs),
                "epoch": epoch,
            }
            accelerator.log(logs, step=epoch)
            progress_bar.close()

            # After each epoch you optionally sample some demo images
            if accelerator.is_main_process:
                is_last_epoch = epoch == config.num_epochs - 1

                if (epoch + 1) % config.save_image_epochs == 0 or is_last_epoch:
                    logger.info("Saving sample predictions...")
                    samples_dir = os.path.join(config.output_dir, "samples")
                    os.makedirs(samples_dir, exist_ok=True)
                    fig = save_samples(
                        accelerator.unwrap_model(model),
                        val_cerra,
                        output_name=f"{samples_dir}/{model_name}_{epoch+1:04d}.png",
                    )
                    if is_last_epoch:
                        tfboard_tracker.writer.add_figure(
                            "Predictions", fig, global_step=epoch
                        )

                if (epoch + 1) % config.save_model_epochs == 0 or is_last_epoch:
                    logger.info("Saving model weights...")
                    model.save_pretrained(config.output_dir)
                    if config.push_to_hub:
                        repo.push_to_hub(
                            commit_message=f"Epoch {epoch+1}", blocking=True
                        )

        return model

    trained_model = inner_training_loop(model)
    accelerator.end_training()

    return trained_model, repo_name
