import os
from datetime import datetime

import click
import pandas as pd
import requests

from deepr.utilities.logger import get_logger

logger = get_logger(__name__)


def download_data(
    variable: str,
    date: str,
    project: str,
    spatial_resolution: str,
    output_directory: str,
) -> None:
    """
    Download data from a given URL to the specified output directory.

    Parameters
    ----------
    variable : str
        The variable name.

    date : str
        The date in the format YYYYMM.

    project : str
        The project name.

    spatial_resolution : str
        The spatial resolution.

    output_directory : str
        The path to the output directory.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the input date is outside the supported ranges.
    """
    date_time = datetime.strptime(date, "%Y%m")

    start_date1 = datetime(1985, 1, 1)
    end_date1 = datetime(2018, 12, 31)

    start_date2 = datetime(2019, 1, 1)
    end_date2 = datetime(2021, 12, 31)

    if start_date1 <= date_time <= end_date1:
        date_range = "1985_2018"
    elif start_date2 <= date_time <= end_date2:
        date_range = "2019_2021"
    else:
        raise ValueError(
            "Invalid date range. "
            "Supported ranges are between 1985 and 2018, or between 2019 and 2021."
        )

    cloud_url = "https://storage.ecmwf.europeanweather.cloud/Code4Earth"
    project_dir = f"netCDF_{project}_{date_range}"
    filename = f"{variable}_{project}_{date}_{spatial_resolution}.nc"
    output_path = os.path.join(output_directory, filename)

    if os.path.exists(output_path):
        logger.info(f"File {output_path} already exists!")
    else:
        response = requests.get(f"{cloud_url}/{project_dir}/{filename}")
        if response.status_code == 200:
            with open(output_path, "wb") as file:
                file.write(response.content)
            logger.info(f"Data downloaded successfully to: {output_path}")
        else:
            logger.info(f"Failed to download data. Status code: {response.status_code}")


@click.command()
@click.argument(
    "output_directory", type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.argument("variable")
@click.argument("project")
@click.argument("spatial_resolution")
@click.argument("start_date")
@click.argument("end_date")
def main(output_directory, variable, project, spatial_resolution, start_date, end_date):
    # Convert start_date and end_date to datetime objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    dates = pd.date_range(start_date, end_date, freq="MS")

    logger.info(
        f"Downloading data to {output_directory} for variable {variable}, "
        f"project {project}, spatial resolution {spatial_resolution} from "
        f"{start_date} to {end_date}"
    )

    for date in dates:
        logger.info(f"Downloading data for date: {date}")
        date_str = date.strftime("%Y%m")
        download_data(variable, date_str, project, spatial_resolution, output_directory)


if __name__ == "__main__":
    main()
