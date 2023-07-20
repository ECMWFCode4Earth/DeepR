import numpy
import pandas
import torch
import tqdm
import xarray

from deepr.model.conditional_ddpm import cDDPMPipeline
from deepr.model.utils import get_hour_embedding


def generate_validation_datasets(
    data_loader, data_scaler_func, model, model_inference_steps, baseline
):
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
    model_inference_steps : int
        Number of inference steps to be used during model prediction. This parameter is
        only relevant if the model is an instance of cDDPMPipeline. Otherwise, it will
        be ignored.
    baseline : str
        The mode used for interpolation during baseline prediction.
        It should be one of the following strings:
        'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', or 'area'.

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
    list_pred_nn = []
    list_pred_base = []
    list_obs = []

    progress_bar = tqdm.tqdm(total=len(data_loader), desc="Batch ")
    # Make predictions for every batch
    for i, (era5, cerra, times) in enumerate(data_loader):
        if isinstance(model, cDDPMPipeline):
            hour_emb = get_hour_embedding(times[:, :1], "class", 24).to(model.device)
            pred_nn = model(
                images=era5,
                class_labels=hour_emb,
                num_inference_steps=model_inference_steps,
                generator=torch.manual_seed(2023),
                output_type="tensor",
            ).images
        else:
            with torch.no_grad():
                pred_nn = model(era5, return_dict=False)[0]

        pred_base = torch.nn.functional.interpolate(
            era5[..., 6:-6, 6:-6], scale_factor=5, mode=baseline
        )

        if data_scaler_func is not None:
            pred_nn = data_scaler_func(pred_nn, times[:, 2])
            pred_base = data_scaler_func(pred_base, times[:, 2])
            cerra = data_scaler_func(cerra, times[:, 2])

        times = transform_times_to_datetime(times)
        latitudes = data_loader.dataset.label_latitudes
        longitudes = data_loader.dataset.label_longitudes
        pred_nn = transform_data_to_xr_format(
            pred_nn, "pred", latitudes, longitudes, times
        )
        pred_base = transform_data_to_xr_format(
            pred_base, "pred_baseline", latitudes, longitudes, times
        )
        obs = transform_data_to_xr_format(cerra, "obs", latitudes, longitudes, times)
        list_pred_nn.append(pred_nn)
        list_pred_base.append(pred_base)
        list_obs.append(obs)
        progress_bar.update(1)

    progress_bar.close()

    pred_nn = xarray.concat(list_pred_nn, dim="time").sortby("time")
    pred_base = xarray.concat(list_pred_base, dim="time").sortby("time")
    obs = xarray.concat(list_obs, dim="time").sortby("time")
    return (
        pred_nn.chunk(chunks={"latitude": 20, "longitude": 40}),
        pred_base.chunk(chunks={"latitude": 20, "longitude": 40}),
        obs.chunk(chunks={"latitude": 20, "longitude": 40}),
    )


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
    data = numpy.asarray(data)

    # Create a dictionary to hold data variables
    data_vars = {varname: (["time", "channel", "latitude", "longitude"], data)}

    # Create coordinate variables
    coords = {"latitude": latitudes, "longitude": longitudes, "time": times}

    # Create the xarray dataset
    dataset = xarray.Dataset(data_vars, coords=coords)

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
