import os
from pathlib import Path

import numpy as np
import pandas
import torch
import tqdm
import xarray as xr
from statsmodels.nonparametric.smoothers_lowess import lowess

from deepr.model.conditional_ddpm import cDDPMPipeline


def generate_validation_dataset(
    data_loader, data_scaler_func, model, config
) -> xr.Dataset:
    """
    Generate validation datasets.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        DataLoader containing the validation data batches.
    data_scaler_func : callable
        Function to scale the model predictions and observed data. It should accept the
        model predictions and observed data tensors along with their corresponding
        times (timestamps) and return the scaled tensors.
        If set to None, no scaling will be applied.
    model : torch.nn.Module
        The machine learning model to be used for predictions.
        The model should either accept images and hour embeddings as inputs
        (if it's an instance of cDDPMPipeline) or just images as inputs.
        The output of the model should be the predicted images.
    config : dict
        Configuration settings for the validation.

    Returns
    -------
    tuple
        pred_nn : xarray.DataArray
            Predicted data in xarray format for the machine learning model.
        pred_base : xarray.DataArray
            Predicted baseline data in xarray format (interpolated using 'baseline')
            for comparison with the model predictions.
        obs : xarray.DataArray
            Observed data in xarray format for validation.
    """
    odir = Path(config["output_dir"]) / config["repo_name"].split("/")[-1]
    os.makedirs(odir, exist_ok=True)

    preds = []
    current_month = None
    progress_bar = tqdm.tqdm(total=len(data_loader), desc="Batch ")
    for i, (era5, cerra, times) in enumerate(data_loader):
        pred_da, times = predict_xr(
            model,
            era5,
            times,
            config,
            # orog_low=orog_low_land,
            # orog_high=orog_high_land,
            data_scaler_func=data_scaler_func,
            latitudes=data_loader.dataset.label_latitudes,
            longitudes=data_loader.dataset.label_longitudes,
        )

        # Save data based on specs
        if config["save_freq"] == "batch":
            filename = "prediction_" + times[0].strftime("%HH_%d-%m-%Y") + ".nc"
            pred_da.to_netcdf(odir / filename)
        elif config["save_freq"] == "month":
            if current_month is None:
                current_month = np.datetime64(pred_da.time.min().values, "M")
            preds.append(pred_da)
            next_month = np.datetime64(pred_da.time.max().values, "M")
            if current_month != next_month:
                pred_ds = xr.concat(preds, dim="time")
                month_pred_ds = pred_ds.where(
                    pred_ds.time.dt.month == current_month.astype(object).month
                ).dropna("time")
                month_pred_ds.to_netcdf(odir / f"prediction_{current_month}.nc")
                current_month = next_month
                next_month_da = pred_ds.where(
                    pred_ds.time.dt.month == next_month.astype(object).month
                ).dropna("time")
                preds = [next_month_da]
            elif config["save_freq"] == "all":
                preds.append(pred_da)
        progress_bar.update(1)
    progress_bar.close()
    if config["save_freq"] == "month":
        pred_ds = xr.concat(preds, dim="time")
        pred_ds.to_netcdf(odir / f"prediction_{current_month}.nc")
    elif config["save_freq"] == "all":
        pred_ds = xr.concat(preds, dim="time")
        d0, d1 = data_loader.dataset.init_date, data_loader.dataset.end_date
        pred_ds.to_netcdf(odir / f"prediction_{d0}-{d1}.nc")


def predict_xr(
    model,
    era5,
    times,
    config,
    orog_low: np.array = None,
    orog_high: np.array = None,
    data_scaler_func=None,
    latitudes: np.array = None,
    longitudes: np.array = None,
):
    if isinstance(model, str):
        prediction = torch.nn.functional.interpolate(
            era5[..., 6:-6, 6:-6],
            mode=model,
            scale_factor=5,
        )

        xs = np.ravel(orog_low)[~np.isnan(np.ravel(orog_low))]
        true_orog = np.ravel(orog_high)
        expected_orog = np.ravel(
            torch.nn.functional.interpolate(
                torch.from_numpy(orog_low[np.newaxis, np.newaxis, ...]),
                mode=model,
                scale_factor=5,
            ).numpy()
        )

        deltas = []
        for i in range(prediction.shape[0]):
            vals = np.ravel(era5[i, 0, 6:-6, 6:-6])[~np.isnan(np.ravel(orog_low))]
            delta = lowess(vals, xs, xvals=expected_orog) - lowess(
                vals, xs, xvals=true_orog
            )
            deltas.append(delta.reshape(1, 1, *prediction.shape[-2:]))
        d = np.nan_to_num(np.concatenate(deltas, axis=0), 0)
        prediction -= d

        attrs = {
            "method": f"{model} + orography correction",
            "orog_correction": "Difference between LOWESS estimates (fitted at each "
            "sample) at low-res & high-res orography.",
        }
    elif isinstance(model, cDDPMPipeline):
        prediction = model(
            images=era5,
            class_labels=times[:, :1].to(model.device),
            eta=config["eta"],
            num_inference_steps=config["inference_steps"],
            generator=torch.manual_seed(config.get("seed", 2023)),
            output_type="tensor",
        ).images
        attrs = {
            "ddpm": model.__class__.__name__,
            "eta": config["eta"],
            "num_inference_steps": config["inference_steps"],
            "seed": config.get("seed", 2023),
        }
    else:
        with torch.no_grad():
            prediction = model(era5, return_dict=False)[0]
        attrs = {}
    attrs["repo"] = config["repo_name"]
    attrs["input_inference_scaling"] = config["inference_scaling"]["input"]
    attrs["output_inference_scaling"] = config["inference_scaling"]["output"]

    if data_scaler_func is not None:
        prediction = data_scaler_func(prediction, times[:, 2])

    times = transform_times_to_datetime(times)
    pred_da = transform_data_to_xr_format(
        prediction, "prediction", latitudes, longitudes, times
    ).chunk(chunks={"latitude": 20, "longitude": 40})
    pred_da.prediction.attrs = attrs

    return pred_da, times


def transform_data_to_xr_format(data, varname, latitudes, longitudes, times):
    """
    Create a xarray dataset from the given variables.

    Parameters
    ----------
    data : numpy array
        The prediction data with shape
        (num_times, num_channels, num_latitudes, num_longitudes).
    varname: str
        Name for the variable
    latitudes : list or numpy array
        List of latitude values.
    longitudes : list or numpy array
        List of longitude values.
    times : list or pandas DatetimeIndex
        List of timestamps.

    Returns
    -------
    xarray.Dataset
        The xarray dataset containing the prediction data.

    Example
    -------
    # Assuming the given variables have appropriate values
    pred_nn = np.random.rand(16, 1, 160, 240)
    latitudes = [44.95, 44.9, ...]  # List of 160 latitude values
    longitudes = [-6.85, -6.8, ...]  # List of 240 longitude values
    times = pd.date_range('2018-01-01', periods=16, freq='3H')
    dataset = create_xarray_dataset(pred_nn, latitudes, longitudes, times)
    print(dataset)
    """
    # Ensure pred_nn is a numpy array
    data = np.asarray(data)

    # Create a dictionary to hold data variables
    data_vars = {varname: (["time", "channel", "latitude", "longitude"], data)}

    # Create coordinate variables
    coords = {"latitude": latitudes, "longitude": longitudes, "time": times}

    # Create the xarray dataset
    dataset = xr.Dataset(data_vars, coords=coords)

    # Remove channel dimension
    dataset = dataset.mean("channel")

    return dataset


def transform_times_to_datetime(times: torch.tensor):
    """
    Transform a tensor of times into a list of datetimes.

    Parameters
    ----------
        times (tensor): A tensor containing times in the format [hour, day, month, year].

    Returns
    -------
        list: A list of pandas datetime objects representing the input times.

    Example:
        times = tensor([[0, 1, 1, 2018],
                        [3, 1, 1, 2018],
                        [6, 1, 1, 2018]])
        result = transform_times_to_datetime(times)
        print(result)
        # Output: [Timestamp('2018-01-01 00:00:00'),
        #          Timestamp('2018-01-01 03:00:00'),
        #          Timestamp('2018-01-01 06:00:00')]
    """
    # Convert each time entry to a pandas datetime object
    datetime_list = [
        pandas.to_datetime(f"{time[3]}-{time[2]}-{time[1]} {time[0]}:00")
        for time in times
    ]

    return datetime_list
