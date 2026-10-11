"""Masked Mean Squared Error loss for Deep Image Prior optimization."""

import torch
import torch.nn as nn
from typing import Optional


class MaskedMSELoss(nn.Module):
    """
    Masked MSE Loss:
      L = || M * (f_theta(z) - y) ||^2 / (sum(M) + eps)
    where:
      - M is binary valid mask (1.0 = valid pixel, 0.0 = damaged/missing)
      - f_theta(z) is restored network output
      - y is degraded input target
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
            if mask.shape[1] == 1 and pred.shape[1] > 1:
                mask = mask.repeat(1, pred.shape[1], 1, 1)
            diff = diff * mask
            valid_pixels = torch.sum(mask) + self.eps
            loss = torch.sum(diff**2) / valid_pixels
        else:
            loss = torch.mean(diff**2)
        return loss


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Functional interface for masked MSE loss."""
    return MaskedMSELoss(eps=eps)(pred, target, mask)
