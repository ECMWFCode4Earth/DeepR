import glob
import os

import click
import xarray

from deepr.utilities.logger import get_logger

logger = get_logger(__name__)


def adjust_latitudes(lat_min, lat_max, lat_values):
    # Check if latitude values are reversed (e.g., from 90 to -90)
    if lat_values[0] > lat_values[-1]:
        lat_min, lat_max = lat_max, lat_min
    return lat_min, lat_max


@click.command()
@click.argument(
    "input_directory", type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.argument("output_directory", type=click.Path(file_okay=False, resolve_path=True))
@click.argument("lon_min", type=float)
@click.argument("lon_max", type=float)
@click.argument("lat_min", type=float)
@click.argument("lat_max", type=float)
def main(input_directory, output_directory, lon_min, lon_max, lat_min, lat_max):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    input_files = glob.glob(f"{input_directory}/*.nc")
    input_files.sort()

    logger.info(f"A list of {len(input_files)} will be transformed.")

    for num, input_file in enumerate(input_files):
        logger.info(f"Processing input_file ({num}): {input_file}")
        input_data = xarray.open_dataset(input_file)

        # Adjust latitude values if needed based on the dataset
        lat_min, lat_max = adjust_latitudes(
            lat_min, lat_max, input_data.latitude.values
        )

        input_data_sel = input_data.sel(
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max),
        )
        output_file = input_file.replace(input_directory, output_directory)
        logger.info(f"Writing processed data to {output_file}")
        input_data_sel.to_netcdf(output_file)


if __name__ == "__main__":
    main()
