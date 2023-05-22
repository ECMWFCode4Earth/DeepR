import torch
import torch.nn.functional as F
from torch import nn


class GeGLU(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
