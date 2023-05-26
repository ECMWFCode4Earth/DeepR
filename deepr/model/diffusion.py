from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from deepr.model.utils import gather


class DenoiseDiffusion:
    """Denoising Diffuision Probabilistic Model.

    Attributes
    ----------
    eps_model : nn.Module
        neural network to predict the noise added in the forward process
    beta : torch.Tensor
        variance of the Isotropic Gaussian distribution used to sample the noise in the
        forward process.
    alpha : torch.Tensor
        1 - beta. These values are useful to sample an arbitrary step in closed form.
    alpha_bar: torch.Tensor
        cumulative product of alpha. 1 - alpha_bar is the variance of the noise to
        sample an arbitrary step.
    n_steps : int
        number of steps of the diffusion process
    sigma2 : torch.Tensor
        The variance of the reverse process. In DDPM, it is equal to beta
    """

    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device):
        """
        Initialize the Denoise Diffusion Model with the provided parameters.

        Arguments
        ---------
            eps_model : nn.Module
                The Neural Network that will predict the noise added in the forward
                process.
            n_steps : int
                The number of steps of the diffusion process.
            device : torch.device
                The device to use. Options: "cpu", 0, 1, 2, ...
        """
        super().__init__()
        self.eps_model = eps_model

        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps

    def q_xt_x0(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Auxiliary function.

        This method computes the mean and variance of Gaussian Distribution to sample xt
        from x0.

        Arguments
        ---------
            x0 : torch.Tensor
                Matrix from initial data distribution.
            t : torch.Tensor
                time step at which to sample xt.

        Returns
        -------
            torch.Tensor : the mean of the distribution = x0 * sqrt(alpha_bar[t])
            torch.Tensor : the variance of the distribution = 1 - alpha_bar[t]
        """
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)

        return mean, var

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None
    ):
        """q(xt|x0).

        This method samples xt from x0, using the closed form of q(xt|x0) ~ N(m, vI)
        with m = x0 * sqrt(alpha_bar[t]) and v = 1 - alpha_bar[t].

        Arguments
        ---------
            x0 : torch.Tensor
                Matrix from initial data distribution.
            t : torch.Tensor
                time step at which to sample xt.
            eps : Optional[torch.Tensor], optional
                Noise to be added. Defaults to None, meaining that the noise is sample
                here.

        Returns
        -------
            torch.Tensor: the matrix corresponding to xt
        """
        if eps is None:
            eps = torch.randn_like(x0)

        mean, var = self.q_xt_x0(x0, t)
        return mean + (var**0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """p(xt-1|xt).

        This method samples xt-1 from xt, using the current NN predictions of the noise.

        Arguments
        ---------
            xt : torch.Tensor
                Matrix from initial data distribution.
            t : torch.Tensor
                time step at which to sample xt.

        Returns
        -------
            torch.Tensor: the matrix corresponding to xt-1 according to the current NN.
        """
        eps_theta = self.eps_model(xt, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha**0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var**0.5) * eps

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """Compute the simplified Loss for training the Diffusion Probabilistic Model.

        Arguments
        ---------
            xt : torch.Tensor
                Matrix from initial data distribution.
            noise : Optional[torch.Tensor], optional
                Noise added during the process. Defaults to None.

        Returns
        -------
            torch.Tensor: mean squared error between the noised added in the forward
                process and the noise predicted by the neural network.
        """
        batch_size = x0.shape[0]
        # Get random t for each sample in the batch
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )

        if noise is None:
            noise = torch.randn_like(x0)

        # Sample noise based on each t
        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t)

        return F.mse_loss(noise, eps_theta)
