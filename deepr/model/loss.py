import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

blur = GaussianBlur(5)
pooling = torch.nn.AvgPool2d(kernel_size=5)


def compute_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Loss terms' computation.

    The first loss term is the L1 loss of the predictions with the references. The 2nd
    loss term refers to the L1 loss between the downsampled prediction and reference.
    Lastly, the 3rd loss term is the L1 loss between the blurred prediction and the
    blurred reference.

    Args:
    ----
        prediction (torch.Tensor): prediction tensor
        target (torch.Tensor):  target tensor

    Returns:
    -------
        torch.Tensor: the 3 loss terms
    """
    l1 = F.l1_loss(prediction, target)
    l1_lowres = F.l1_loss(pooling(prediction), pooling(target))
    l1_blur = F.l1_loss(blur(prediction), blur(target))
    return l1, l1_lowres, l1_blur
