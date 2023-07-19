import pandas
import xarray


def compute_climatology_by_time_group(
    data_files: list, group_by: list, output_directory: str
):
    """
    Compute statistics (mean and standard deviation) by time group and save to netCDF.

    Parameters
    ----------
    data_files : list
        A list of file paths containing the data.
    group_by: list
        A list of time components to make the aggregation.
    output_directory : str
        The directory where the output NetCDF files will be saved.

    Returns
    -------
    tuple
        A tuple containing the mean and standard deviation DataArrays computed by
        time group.
    """
    # Open the multi-file dataset
    ds = xarray.open_mfdataset(data_files)

    # Create a MultiIndex for the time components given by group_by
    time_group_idx = pandas.MultiIndex.from_arrays(
        [ds[f"time.{x}"].values for x in group_by]
    )

    # Assign time_group coordinate to the dataset
    ds.coords["time_group"] = ("time", time_group_idx)

    # Compute mean and standard deviation by time_group
    mean_by_group = ds.groupby("time_group").mean()
    mean_by_group.load()
    std_by_group = ds.groupby("time_group").std()
    std_by_group.load()

    # Save mean climatology as NetCDF files
    for time_group in list(mean_by_group.time_group.values):
        time_group_str = "_".join(
            [f"{group_by[i]}-{time_group[i]}" for i in range(len(group_by))]
        )
        selection = mean_by_group.sel(time_group=time_group)
        selection = selection.drop(
            ["time_group"] + [f"time_level_{i}" for i in range(len(group_by))]
        )
        for varname in list(selection.data_vars):
            filename = f"{output_directory}/{varname}_clim-mean_{time_group_str}.nc"
            selection.to_netcdf(filename)

    # Save standard deviation climatology as NetCDF files
    for time_group in list(std_by_group.time_group.values):
        time_group_str = "_".join(
            [f"{group_by[i]}-{time_group[i]}" for i in range(len(group_by))]
        )
        selection = std_by_group.sel(time_group=time_group)
        selection = selection.drop(
            ["time_group"] + [f"time_level_{i}" for i in range(len(group_by))]
        )
        for varname in list(selection.data_vars):
            filename = f"{output_directory}/{varname}_clim-std_{time_group_str}.nc"
            selection.to_netcdf(filename)

    # Return mean and standard deviation DataArrays
    return mean_by_group, std_by_group
