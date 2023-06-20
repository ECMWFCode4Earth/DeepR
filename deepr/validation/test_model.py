import evaluate
import torch


def test_model(
    model,
    dataset: torch.utils.data.IterableDataset,
    hparams: dict = {},
    push_to_hub: bool = False,
    batch_size: int = 16,
):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, pin_memory=True)

    mse = evaluate.load("mse", "multilist")
    for era5, cerra, *times in dataloader:
        # Predict the noise residual
        with torch.no_grad():
            pred = model(era5, return_dict=False)[0]
            mse.add_batches(
                references=cerra.reshape((cerra.shape[0], -1)),
                predictions=pred.reshape((pred.shape[0], -1)),
            )

    test_mse = mse.compute()
    tf_writter.add_hparams(hparams, {"test_mse": val_mse["mse"]})

    if push_to_hub:
        evaluate.save(
            config.output_dir,
            experiment="Train Neural Network",
            **test_mse,
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
