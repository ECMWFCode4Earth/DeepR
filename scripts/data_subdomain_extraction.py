import glob
import os

import xarray

input_directory = (
    "/gpfs/projects/meteo/WORK/PROYECTOS/2023_ecmwfcode4earth/DeepR/data/C3S-CDS/CERRA"
)
output_directory = (
    "/gpfs/projects/meteo/WORK/PROYECTOS/2023_ecmwfcode4earth/DeepR/data/"
    "C3S-CDS/CERRA/subdomain"
)
os.makedirs(output_directory, exist_ok=True)

spatial_coverage = {"latitude": [44.95, 37], "longitude": [-6.85, 5.1]}

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
