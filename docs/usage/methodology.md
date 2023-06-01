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

### U-Net

In particular, a tailored U-Net architecture with 2D convolutions, residual connections and attetion layers is used.

![U-Net Architecture Diagram](./docs/_media/eps-U-Net%20diagram.svg)

The parameteres of these model implemented in [deepr/model/unet.py](deepr/model/unet.py) are:

- `image_channels`: It is the number of channels of the high resolution imagen we want to generate, that matches with the number of channels of the output from the U-Net. Default value is `1`, as we plan to sample one variable at a time.

- `n_channels`: It is the number of output channels of the initial Convolution. Defaults to `16`.

- `channel_multipliers`: It is the multiplying factor over the channels applied at each down/upsampling level of the U-Net. Defaults to `[1, 2, 2, 4]`.

- `is_attention`: It represents the use of Attention over each down/upsampling level of the U-Net. Defaults to `[False, False, True, True]`.

- `n_blocks`: The number of residual blocks considered in each level. Defaults to `2`.

- `conditioned_on_input`: The number of channels of the conditions considered.

*NOTE I*: The length of `channel_multipliers` and `is_attention` should match as it sets the number of resolutions of our U-Net architecture.

*NOTE II*: Spatial tensors fed to Diffusion model must have shapes of length multiple of $2^{\\text{num resolutions} - 1}$.

#### Downsampling

#### Upsampling

#### Down Block

#### Up Block

#### Residual Block

#### Final Block
