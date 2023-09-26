import calendar
import os
import pathlib

import cdsapi
import click
import numpy
import pandas
import xarray

from deepr.utilities.logger import get_logger

logger = get_logger(__name__)


def get_number_of_days(year: int, month: int) -> int:
    """
    Get the number of days in a specific month and year.

    Parameters
    ----------
    year : int
        The year.
    month : int
        The month (1 to 12).

    Returns
    -------
    int
        The number of days in the specified month and year, or None if the month or
        year is invalid.
    """
    try:
        _, num_days = calendar.monthrange(year, month)
        return num_days
    except ValueError:
        return None  # Invalid month or year


def download_cds_data(
    output_directory: str, variable: str, month: int, year: int
) -> tuple:
    """
    Download data from CDS (Climate Data Store) and save it to the specified directory.

    Parameters
    ----------
    output_directory : str
        The path to the output directory.
    variable : str
        The variable name.
    month : int
        The month (1 to 12).
    year : int
        The year.

    Returns
    -------
    tuple
        A tuple containing the paths to the downloaded ERA5 and CERRA data files.
    """
    c = cdsapi.Client()
    cds_variable = {
        "t2m": "2m_temperature",
    }

    # Create directories if they don't exist
    path_cerra = pathlib.Path(
        f"{output_directory}/labels/{variable}/"
        f"{variable}_cerra_{year}{'{:02d}'.format(month)}_005deg.nc"
    )
    if not path_cerra.parent.exists():
        path_cerra.parent.mkdir(exist_ok=True, parents=True)

    logger.info(f"Downloading CERRA data to {path_cerra}")
    c.retrieve(
        "reanalysis-cerra-single-levels",
        {
            "format": "netcdf",
            "variable": cds_variable[variable],
            "level_type": "surface_or_atmosphere",
            "data_type": "reanalysis",
            "product_type": "analysis",
            "year": year,
            "month": "{:02d}".format(month),
            "day": [
                "{:02d}".format(x)
                for x in range(1, get_number_of_days(year, month) + 1)
            ],
            "time": ["{:02d}:00".format(x) for x in range(0, 24, 3)],
        },
        path_cerra,
    )

    path_era5 = pathlib.Path(
        f"{output_directory}/features/{variable}/"
        f"{variable}_era5_{year}{'{:02d}'.format(month)}_025deg.nc"
    )
    if not path_era5.parent.exists():
        path_era5.parent.mkdir(exist_ok=True, parents=True)

    logger.info(f"Downloading ERA5 data to {path_era5}")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": cds_variable[variable],
            "year": year,
            "month": "{:02d}".format(month),
            "day": [
                "{:02d}".format(x)
                for x in range(1, get_number_of_days(year, month) + 1)
            ],
            "time": ["{:02d}:00".format(x) for x in range(0, 24, 3)],
            "format": "netcdf",
        },
        path_era5,
    )
    return path_era5, path_cerra


def process_cds_data(file_path: pathlib.Path, varname: str) -> pathlib.Path:
    """
    Process CDS data file by adjusting longitudes and data type.

    Parameters
    ----------
    file_path : pathlib.Path
        The path to the CDS data file.
    varname : str
        The variable name.

    Returns
    -------
    pathlib.Path
        The path to the processed CDS data file.
    """
    ds = xarray.open_dataset(file_path)
    lon = ds["longitude"]

    # Adjust longitudes
    ds["longitude"] = ds["longitude"].where(lon <= 180, other=lon - 360)
    ds = ds.reindex(**{"longitude": sorted(ds["longitude"])})

    # Modify encoding attributes
    encoding_attrs = ds[varname].encoding.copy()
    del encoding_attrs["scale_factor"]
    del encoding_attrs["add_offset"]
    encoding_attrs["dtype"] = numpy.float64
    ds[varname].encoding = encoding_attrs

    # Save the processed data to a new file
    new_file = file_path.with_suffix("_new.nc")
    ds.to_netcdf(new_file)

    # Remove the original file and rename the new file
    os.remove(file_path)
    os.rename(new_file, file_path)

    return file_path


@click.command()
@click.argument("output_directory", type=click.Path(file_okay=False, resolve_path=True))
@click.argument("variable", type=click.Choice(["t2m"]))
@click.argument("start_date", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.argument("end_date", type=click.DateTime(formats=["%Y-%m-%d"]))
def main(output_directory, variable, start_date, end_date):
    """
    Download CDS data for a specific variable within the given date range and save it to the output directory.

    Parameters
    ----------
    output_directory : str
        The path to the output directory.
    variable : str
        The variable name (e.g., "t2m").
    start_date : datetime
        The start date in YYYY-MM-DD format.
    end_date : datetime
        The end date in YYYY-MM-DD format.
    """
    dates = pandas.date_range(start_date, end_date, freq="MS")
    features, labels = [], []
    for date in dates:
        data_feature, data_label = download_cds_data(
            output_directory, variable, date.year, date.month
        )
        data_feature = process_cds_data(data_feature, variable)
        data_label = process_cds_data(data_label, variable)
        features.append(data_feature)
        labels.append(data_label)
    print(features)
    print(labels)


if __name__ == "__main__":
    main()
