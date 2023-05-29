import requests
import os
from datetime import datetime


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
    date = datetime.strptime(date, "%Y%m")

    start_date1 = datetime(1985, 1, 1)
    end_date1 = datetime(2018, 12, 31)

    start_date2 = datetime(2019, 1, 1)
    end_date2 = datetime(2021, 12, 31)

    if start_date1 <= date <= end_date1:
        date_range = "1985_2018"
    elif start_date2 <= date <= end_date2:
        date_range = "2019_2021"
    else:
        raise ValueError(
            "Invalid date range. "
            "Supported ranges are between 1985 and 2018, or between 2019 and 2021."
        )

    url = f"https://storage.ecmwf.europeanweather.cloud/Code4Earth/" \
          f"netCDF_{project}_{date_range}/" \
          f"{variable}_{project}_{date.strftime('%Y%m')}_{spatial_resolution}.nc"
    filename = f"{variable}_{project}_{date.strftime('%Y%m')}_{spatial_resolution}.nc"
    output_path = os.path.join(output_directory, filename)

    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, "wb") as file:
            file.write(response.content)
        print(f"Data downloaded successfully to: {output_path}")
    else:
        print(f"Failed to download data. Status code: {response.status_code}")


if __name__ == '__main__':
    import pandas
    output_directory = "/predictia-nas2/Data/DeepR/"
    variable = "t2m"
    project = "era5"
    spatial_resolution = "025deg"

    start_date = datetime(1985, 1, 1)
    end_date = datetime(2021, 12, 31)
    dates = pandas.date_range(start_date, end_date, freq="MS")

    current_date = start_date
    print(
        f"Downloading data to {output_directory} for variable {variable}, "
        f"project {project}, spatial resolution {spatial_resolution} from "
        f"{start_date} to {end_date}"
    )
    for date in dates:
        print(f"Downloading data for date: {date}")
        date_str = date.strftime("%Y%m")
        download_data(variable, date_str, project, spatial_resolution, output_directory)


