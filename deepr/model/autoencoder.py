import torch
from torch import nn

from deepr.model.decoder import Decoder
from deepr.model.encoder import Encoder
from deepr.model.utils import GaussianDistribution


class Autoencoder(nn.Module):
    def __init__(
        self, encoder: Encoder, decoder: Decoder, emb_channels: int, z_channels: int
    ):
        """
        Initialize an autoencoder model given some parameters.

        The parameters provided are:
        encoder, decoder, embedding channels, and z channels.

        Parameters
        ----------
        encoder : Encoder
            An instance of the Encoder class.
        decoder : Decoder
            An instance of the Decoder class.
        emb_channels : int
            An integer representing the number of embedding channels.
        z_channels : int
            An integer representing the number of z channels.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # Convolution to map from embedding space to quantized embedding space moments
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * emb_channels, 1)
        # Convolution to map from quantized embedding space back to embedding space
        self.post_quant_conv = nn.Conv2d(emb_channels, z_channels, 1)

    def encode(self, img: torch.Tensor) -> GaussianDistribution:
        # Get embeddings with shape `[batch_size, z_channels * 2, z_height, z_height]`
        z = self.encoder(img)
        # Get the moments in the quantized embedding space
        moments = self.quant_conv(z)
        # Return the distribution
        return GaussianDistribution(moments)

    def decode(self, z: torch.Tensor):
        # Map to embedding space from the quantized representation
        z = self.post_quant_conv(z)
        # Decode the image of shape `[batch_size, channels, height, width]`
        return self.decoder(z)
