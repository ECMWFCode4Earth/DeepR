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
wget -O data.zip https://cloud.predictia.es/s/zen8PGwJbi7mTCB/download
unzip data.zip
rm data.zip
```

Before pushing to GitHub, run the following commands:

1. Update conda environment: `make conda-env-update`
1. Install this package: `pip install -e .`
1. Sync with the latest [template](https://github.com/ecmwf-projects/cookiecutter-conda-package) (optional): `make template-update`
1. Run quality assurance checks: `make qa`
1. Run tests: `make unit-tests`
1. Run the static type checker: `make type-check`
1. Build the documentation (see [Sphinx tutorial](https://www.sphinx-doc.org/en/master/tutorial/)): `make docs-build`

## Data

### Data specifications

The spatial coverage of the datasets provided is described below:

Features: (240, 150) ------- Label: (800, 500)

```complete-spatial-coverage.yml
data_configuration:
  features_configuration:
    spatial_coverage:
      longitude: [ -20.5, 39.25 ]
      latitude: [ 66.25, 29 ]
  label_configuration:
    spatial_coverage:
      longitude: [ -10, 29.95]
      latitude: [ 60, 35.05 ]
```

During the development stage, a subset of the data is used to validate the implementation of the model:

Features: (32, 20)------- Label: (32, 20)

```reduce-spatial-coverage.yml
data_configuration:
  features_configuration:
    spatial_coverage:
      longitude: [ 6.0, 13.75 ]
      latitude: [ 50, 45.25 ]
  label_configuration:
    spatial_coverage:
      longitude: [ 9.2, 10.75]
      latitude: [ 48, 47.05 ]
```

## Methodology

The main purpose of this library is to test the capabilities of deep diffusion models for reanalysis super-resolution tasks.

The objectives of this challenge focus on:

- Explore the capabilities of Deep Diffusion models to represent high resolution reanalysis datasets.

- Evaluate the impact of including several covariables in the model.

  - Conditioning on time stamps
  - Conditioning on meteorological covariables
  - Conditioning on in-site observations

### Super Resolution Diffusion model

Explanation of Diffusion processes...


Here, DL is considered to model the $\eps_t$ sampled at each step given $x_{t+1}$ and conditioned on the LR image. 

#### Training

During training, for each batch of data, we sample random timesteps $t$ and noise $\eps_{t}$ and derive the corresponding values $x_t$. Then, we train our DL model to minimize the following loss function:

$$ \mathcal{L} (x) = || $\eps_{t}$ - \Phi \left(x_{t+1}, t \right) ||^2$$

which is the mean squared error (MSE) between:
- the noise, $\eps_{t}$, added at timestep $t$

- the prediction of the DL model, $\Phi$, taking as input the timestep $t$ and the noisy matrix $x_{t+1}$.


#### Inference

During inference, we can sample random noise and run the reverse process conditioned on input ERA5 grids, to obtain high resolution reanalysis grids. Another major benefit from this approach is the possibility of generation an ensemble of grids to represent its uncertainty avoiding the mode collapse (common in GANs). 


### U-Net

In particular, a tailored U-Net architecture with 2D convolutions, residual connections and attetion layers is used. 


![U-Net Architecture Diagram](docs/_media/eps-U-Net%20diagram.svg)


The parameteres of these model implemented in 'deepr/model/unet.py' are:

- 'image_channels': It is the number of channels of the high resolution imagen we want to generate, that matches with the number of channels of the output from the U-Net. Default value is `1`, as we plan to sample one variable at a time.

- 'n_channels': It is the number of output channels of the initial Convolution. Defaults to `16`.

- 'channel_multipliers': It is the multiplying factor over the channels applied at each down/upsampling level of the U-Net. Defaults to `[1, 2, 2, 4]`.

- 'is_attention': It represents the use of Attention over each down/upsampling level of the U-Net. Defaults to `[False, False, True, True]`.

- 'n_blocks': The number of residual blocks considered in each level. Defaults to `2`.

- 'conditioned_on_input': The number of channels of the conditions considered.

NOTE I: The length of 'channel_multipliers' and 'is_attention' should match as it sets the number of resolutions of our U-Net architecture.
NOTE II: Spatial tensors fed to Diffusion model must have shapes of length multiple of $2^{(\# resolutions) - 1}$.


#### Downsampling


#### Upsampling


#### Down Block


#### Up Block


#### Residual Block


#### Final Block



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
