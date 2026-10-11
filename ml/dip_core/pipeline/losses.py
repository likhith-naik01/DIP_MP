import torch
import torch.nn as nn
from typing import Optional


class MaskedMSELoss(nn.Module):
    """
    Masked Mean Squared Error Loss:
      L = mean((M * (f_θ(z) - y))^2)
    normalized over the valid mask pixels (where M == 1).

    Args:
      eps: Numerical stability constant.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = pred - target
        if mask is not None:
            # Broadcast mask if channels differ
            if mask.shape[1] == 1 and pred.shape[1] > 1:
                mask = mask.repeat(1, pred.shape[1], 1, 1)
            diff = diff * mask
            valid_pixels = torch.sum(mask) + self.eps
            loss = torch.sum(diff**2) / valid_pixels
        else:
            loss = torch.mean(diff**2)
        return loss
