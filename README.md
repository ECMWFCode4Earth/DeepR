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

Features: (240, 150) ------- Label: (800, 480)

```complete-spatial-coverage.yml
data_configuration:
  features_configuration:
    spatial_coverage:
      longitude: [ -20.5, 39.25 ]
      latitude: [ 66.25, 29 ]
  label_configuration:
    spatial_coverage:
      longitude: [ -10, 29.95]
      latitude: [ 59.6, 35.65 ]
```

During the development stage, a subset of the data is used to validate the implementation of the model:

Features: (32, 24)------- Label: (32, 24)

```reduce-spatial-coverage.yml
data_configuration:
  features_configuration:
    spatial_coverage:
      longitude: [ 6.0, 13.75 ]
      latitude: [ 50.25, 44.5 ]
  label_configuration:
    spatial_coverage:
      longitude: [ 9.2, 10.75]
      latitude: [ 48.1, 46.95 ]
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

Here, DL is considered to model the $\\epsilon_t$ sampled at each step given $x\_{t+1}$ and conditioned on the LR image.

#### Training

During training, for each batch of data, we sample random timesteps $t$ and noise $\\epsilon\_{t}$ and derive the corresponding values $x_t$. Then, we train our DL model to minimize the following loss function:

$$ \\mathcal{L} (x) = || \\epsilon\_{t} - \\Phi \\left(x\_{t+1}, t \\right) ||^2$$

which is the mean squared error (MSE) between:

- the noise, $\\epsilon\_{t}$, added at timestep $t$

- the prediction of the DL model, $\\Phi$, taking as input the timestep $t$ and the noisy matrix $x\_{t+1}$.

#### Inference

During inference, we can sample random noise and run the reverse process conditioned on input ERA5 grids, to obtain high resolution reanalysis grids. Another major benefit from this approach is the possibility of generation an ensemble of grids to represent its uncertainty avoiding the mode collapse (common in GANs).

### $\\epsilon\_{t}$-model

The library [diffusers](https://huggingface.co/docs/diffusers/v0.16.0/en/api/models) brings several options to include in Diffusion Models. Here, we present several of the options included there that may fit our use case as well as our own tailored implementations.

#### diffusers.UNet2DModel

The [diffusers.UNet2DModel](https://huggingface.co/docs/diffusers/v0.16.0/en/api/models#diffusers.UNet2DModel) is the most similar class to our implementation, which is a U-net architecture with several options for down and up blocks.

- **Down blocks**: DownBlock2D, ResnetDownsampleBlock2D, AttnDownBlock2D, CrossAttnDownBlock2D, SimpleCrossAttnDownBlock2D, SkipDownBlock2D, AttnSkipDownBlock2D, DownEncoderBlock2D, AttnDownEncoderBlock2D, KDownBlock2D and KCrossAttnDownBlock2D.

- **Up block**: UpBlock2D, ResnetUpsampleBlock2D, CrossAttnUpBlock2D, SimpleCrossAttnUpBlock2D, AttnUpBlock2D, SkipUpBlock2D, AttnSkipUpBlock2D, UpDecoderBlock2D, AttnUpDecoderBlock2D, KUpBlock2D and KCrossAttnUpBlock2D.

One example configuration to use [diffusers.UNet2DModel](https://huggingface.co/docs/diffusers/v0.16.0/en/api/models#diffusers.UNet2DModel) is included below:

```
training_configuration:
  ...
  model_configuration:
    eps_model:
      class_name: diffusers.UNet2DModel
      kwargs:
        block_out_channels: [32, 64, 128]
        down_block_types: [DownBlock2D, AttnDownBlock2D, AttnDownBlock2D]
        up_block_types: [AttnUpBlock2D, AttnUpBlock2D, UpBlock2D]
        layers_per_block: 2
        time_embedding_type: positional
        in_channels: 2
        out_channels: 1
        sample_size: [20, 32]
  ...
```

The [diffusers.UNet2DModel](https://huggingface.co/docs/diffusers/v0.16.0/en/api/models#diffusers.UNet2DModel) also accepts conditioning on labels through its argument `class_labels`. First, the embedding type must be specified in the [`__init__`](https://github.com/huggingface/diffusers/blob/v0.16.0/src/diffusers/models/unet_2d.py#L83) method trough:

- Passing `class_embed_type` (Options are 'timestep', 'identity' or None).
- Passing `num_class_embeds` with the size of the dictionary of embeddings to use.

For example, to consider the hour of the data as covariate in this model we have two options:

**Option A:** Set `num_class_embeds = 24` in the model creation and `hour_embed_type = class` in training configuration. This way the model learns a Embedding table for each hour.

**Option B:** Set `class_embed_type = identity` in the model configuration and `hour_embed_type = positional` in training configuration.

**Option C:** Set `class_embed_type = timestep` in the model configuration and `hour_embed_type` = `timestep` in training configuration. This configuration applies the same cos & sin transformation as in Option B maintaining the same `max_duration=10000`. Unlike Option B, we fit 2 `nn.Linear` after the embedding before feeding it to the NN.

#### diffusers.UNet2DConditionModel

The[diffusers.UNet2DConditionModel](https://huggingface.co/docs/diffusers/v0.16.0/en/api/models#diffusers.UNet2DConditionModel) is an extension of the previous [diffusers.UNet2DModel](https://huggingface.co/docs/diffusers/v0.16.0/en/api/models#diffusers.UNet2DModel) to consider conditions during the reverse process such as time stamps, or other covariables.

One interesting parameter to tune is the activation funcion used in the time embedding which can be: Swish, Mish, SiLU or GELU.

But the most remarkable difference is the possibility of conditioning the reverse diffusion process in the encoder hidden states (comming from images, text, or any other)

One example configuration to use [diffusers.UNet2DConditionModel](https://huggingface.co/docs/diffusers/v0.16.0/en/api/models#diffusers.UNet2DConditionModel) is included below:

```
training_configuration:
  ...
  model_configuration:
    eps_model:
      class_name: diffusers.UNet2DConditionModel
      kwargs:
        block_out_channels: [124, 256, 512]
        down_block_types: [CrossAttnDownBlock2D, CrossAttnDownBlock2D, DownBlock2D]
        mid_block_type: UNetMidBlock2DCrossAttn
        up_block_types: [UpBlock2D, CrossAttnUpBlock2D, CrossAttnUpBlock2D]
        layers_per_block: 2
        time_embedding_type: positional
        in_channels: 2
        out_channels: 1
        sample_size: [20, 32]
        only_cross_attention: False
        cross_attention_dim: 256
        addition_embed_type: other
  ...
```

#### Tailored UNet

In particular, a tailored U-Net architecture with 2D convolutions, residual connections and attetion layers is used.

![U-Net Architecture Diagram](docs/_static/eps-U-Net%20diagram.svg)

The parameteres of these model implemented in [deepr/model/unet.py](deepr/model/unet.py) are:

- `image_channels`: It is the number of channels of the high resolution imagen we want to generate, that matches with the number of channels of the output from the U-Net. Default value is `1`, as we plan to sample one variable at a time.

- `n_channels`: It is the number of output channels of the initial Convolution. Defaults to `16`.

- `channel_multipliers`: It is the multiplying factor over the channels applied at each down/upsampling level of the U-Net. Defaults to `[1, 2, 2, 4]`.

- `is_attention`: It represents the use of Attention over each down/upsampling level of the U-Net. Defaults to `[False, False, True, True]`.

- `n_blocks`: The number of residual blocks considered in each level. Defaults to `2`.

- `conditioned_on_input`: The number of channels of the conditions considered.

*NOTE I*: The length of `channel_multipliers` and `is_attention` should match as it sets the number of resolutions of our U-Net architecture.

*NOTE II*: Spatial tensors fed to Diffusion model must have shapes of length multiple of $2^{\\text{num resolutions} - 1}$.

An example configuration for this model is specified in training_configuration > model_configuration > eps_model,

```
training_configuration:
  ...
  model_configuration:
    eps_model:
      class_name: UNet
      kwargs:
        block_out_channels: [32, 64, 128, 256]
        is_attention: [False, False, True, True]
        layers_per_block: 2
        time_embedding_type: positional
        in_channels: 2
        out_channels: 1
        sample_size: [20, 32]
  ...
```

##### Downsampling

The class [Downsample](deepr/model/unet_blocks.py#LL20) ...

##### Upsampling

The class [Upsample](deepr/model/unet_blocks.py#LL8) ...

##### Down Block

The class [Down block](deepr/model/unet_blocks.py#LL30)

##### Middle Block

The class [Middle block](deepr/model/unet_blocks.py#LL123)

##### Up Block

The class [Up block](deepr/model/unet_blocks.py#LL73)

##### Residual Block

##### Final Block

## Project Outputs

### Models (HuggingFace)

- [**Swin2SR (x4)**](https://huggingface.co/predictia/europe_reanalysis_downscaler_swin2sr): A novel transformed-based method for image super-resolution trained with meteorological datasets.

- [**Conditioned Denoising Diffusion Probabilistic Model**](predictia/europe_reanalysis_downscaler_diffuser): A tailored DDPM that accepts temporal covariates as the hours of the day or the day of the years as input to the $\\eps$-model.

## Appendix I: Positional Embeddings

When working with sequential data, the order of the elements is important, and we must pay attention to how we pass this information to our models.

In our particular case, the timesteps $t$ is encoded with positional embeddings as proposed in the [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf) paper.

![Positional Embeddings](docs/_static/pos_embedding.png)

Besides, we may encoded other important features as the hour of the day or the day of the year, which are cyclical. This is different from positional encodings because we want the encoding from hour 23 to be more similar to the one from 0 than from hour 18.

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising diffusion probabilistic models](https://arxiv.org/pdf/2006.11239.pdf). Advances in Neural Information Processing Systems, 33, 6840-6851.

- Song, J., Meng, C., & Ermon, S. (2020). [Denoising diffusion implicit models](https://arxiv.org/pdf/2010.02502.pdf). arXiv preprint arXiv:2010.02502.

- Conde, M. V., Choi, U. J., Burchi, M., & Timofte, R. (2022). [Swin2SR: Swinv2 transformer for compressed image super-resolution and restoration](https://arxiv.org/pdf/2209.11345.pdf). arXiv preprint arXiv:2209.11345.

- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). [High-resolution image synthesis with latent diffusion models](https://arxiv.org/pdf/2112.10752.pdf). In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10684-10695).

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
