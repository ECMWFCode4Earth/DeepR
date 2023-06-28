import glob
import os

import xarray

input_directory = "/input_directory"
output_directory = "/output_directory"
os.makedirs(output_directory, exist_ok=True)

spatial_coverage = {"latitude": [46.45, 35.50], "longitude": [-8.35, 6.6]}

input_files = glob.glob(f"{input_directory}/*.nc")
input_files.sort()

for input_file in input_files:
    print(f"Processing input_file: {input_file}")
    input_data = xarray.open_dataset(input_file)
    input_data_sel = input_data.sel(
        latitude=slice(
            spatial_coverage["latitude"][0],
            spatial_coverage["latitude"][1],
        ),
        longitude=slice(
            spatial_coverage["longitude"][0],
            spatial_coverage["longitude"][1],
        ),
    )
    output_file = input_file.replace(input_directory, output_directory)
    print(f"Writing processed data to {output_file}")
    input_data_sel.to_netcdf(output_file)
