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


class CompositeEdgeLoss(nn.Module):
    """
    Edge-Preserving and Detail-Enhancing Composite Loss:
      L = L_charbonnier + alpha_grad * L_sobel + alpha_lap * L_laplacian

    Properties:
      - Charbonnier Smooth L1 prevents blurry pixel averaging.
      - Sobel Gradient loss aligns sharp edges and temperature transitions.
      - Laplacian loss enforces high-frequency texture preservation.
      - Fully supports spatial defect masks M in [0, 1].
    """

    def __init__(
        self,
        charb_eps: float = 1e-3,
        alpha_grad: float = 0.5,
        alpha_lap: float = 0.3,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.charb_eps = charb_eps
        self.alpha_grad = alpha_grad
        self.alpha_lap = alpha_lap
        self.eps = eps

        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 8.0
        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)

        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.register_buffer("lap", lap)

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

            diff_masked = diff * mask
            valid_pixels = torch.sum(mask) + self.eps

            # 1. Charbonnier loss (smooth L1)
            charb = torch.sum(torch.sqrt(diff_masked**2 + self.charb_eps**2)) / valid_pixels

            # 2. Gradient difference on masked inputs
            pred_m = pred * mask
            tgt_m = target * mask
            gx_p = torch.nn.functional.conv2d(pred_m, self.kx, padding=1)
            gx_t = torch.nn.functional.conv2d(tgt_m, self.kx, padding=1)
            gy_p = torch.nn.functional.conv2d(pred_m, self.ky, padding=1)
            gy_t = torch.nn.functional.conv2d(tgt_m, self.ky, padding=1)

            grad_diff = (torch.abs(gx_p - gx_t) + torch.abs(gy_p - gy_t)) * mask
            grad_loss = torch.sum(grad_diff) / valid_pixels

            # 3. High-frequency Laplacian difference
            lap_p = torch.nn.functional.conv2d(pred_m, self.lap, padding=1)
            lap_t = torch.nn.functional.conv2d(tgt_m, self.lap, padding=1)
            lap_loss = torch.sum(torch.abs(lap_p - lap_t) * mask) / valid_pixels

            return charb + self.alpha_grad * grad_loss + self.alpha_lap * lap_loss
        else:
            charb = torch.mean(torch.sqrt(diff**2 + self.charb_eps**2))

            gx_p = torch.nn.functional.conv2d(pred, self.kx, padding=1)
            gx_t = torch.nn.functional.conv2d(target, self.kx, padding=1)
            gy_p = torch.nn.functional.conv2d(pred, self.ky, padding=1)
            gy_t = torch.nn.functional.conv2d(target, self.ky, padding=1)
            grad_loss = torch.mean(torch.abs(gx_p - gx_t) + torch.abs(gy_p - gy_t))

            lap_p = torch.nn.functional.conv2d(pred, self.lap, padding=1)
            lap_t = torch.nn.functional.conv2d(target, self.lap, padding=1)
            lap_loss = torch.mean(torch.abs(lap_p - lap_t))

            return charb + self.alpha_grad * grad_loss + self.alpha_lap * lap_loss


def composite_edge_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Functional interface for composite edge-preserving restoration loss."""
    return CompositeEdgeLoss()(pred, target, mask)
