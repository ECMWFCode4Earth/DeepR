# DeepR: Deep Reanalysis.

Global reanalysis downscaling to regional scales by means of deep learning techniques.

## Description

![project-photo-description.png](docs%2F_static%2Fproject-photo-description.png)

## Workflow for developers/contributors

For best experience create a new conda environment (e.g. DEVELOP) with Python 3.10:

```
conda create -n DEVELOP -c conda-forge python=3.10
conda activate DEVELOP
```

A data directory for the testing data must be created:

```
cd tests
mkdir data
cd data
mkdir features
mkdir labels
```

Once the directories have been created, testing data can be downloaded:

```
cd tests
cd data
cd features
wget https://storage.ecmwf.europeanweather.cloud/Code4Earth/t2m_era5_201801_025deg.nc
cd ..
cd labels
wget https://storage.ecmwf.europeanweather.cloud/Code4Earth/t2m_cerra_201801_005deg.nc
```

Before pushing to GitHub, run the following commands:

1. Update conda environment: `make conda-env-update`
1. Install this package: `pip install -e .`
1. Sync with the latest [template](https://github.com/ecmwf-projects/cookiecutter-conda-package) (optional): `make template-update`
1. Run quality assurance checks: `make qa`
1. Run tests: `make unit-tests`
1. Run the static type checker: `make type-check`
1. Build the documentation (see [Sphinx tutorial](https://www.sphinx-doc.org/en/master/tutorial/)): `make docs-build`

## Methodology

The main purpose of this library is to test the capabilities of deep diffusion models for reanalysis super-resolution tasks.

## References

- [Annotated Deep Learning Paper implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations)

## License

```
Copyright 2023, European Union.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
