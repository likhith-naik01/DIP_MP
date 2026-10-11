"""Unit tests verifying mathematical behavior of Masked MSE loss."""

import torch
from dip_core.pipeline.masked_loss import MaskedMSELoss


def test_masked_loss_ignores_masked_regions():
    """Verify changing values in masked regions (M=0) has zero effect on loss."""
    criterion = MaskedMSELoss()

    pred = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
    target1 = torch.tensor([[[[0.5, 0.5], [0.5, 0.0]]]])
    target2 = torch.tensor([[[[0.5, 0.5], [0.5, 999.0]]]])  # Wildly different at [1, 1]
    mask = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]])      # Missing at [1, 1]

    loss1 = criterion(pred, target1, mask)
    loss2 = criterion(pred, target2, mask)

    assert abs(loss1.item() - loss2.item()) < 1e-6
    assert abs(loss1.item()) < 1e-6


def test_masked_loss_gradient_zero_on_missing_pixels():
    """Verify backpropagation gradients are strictly zero on missing pixels."""
    criterion = MaskedMSELoss()

    pred = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], requires_grad=True)
    target = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    mask = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]])  # [1, 1] is masked out

    loss = criterion(pred, target, mask)
    loss.backward()

    # Gradient must be non-zero on valid pixels, but exactly zero on masked pixel
    assert pred.grad[0, 0, 0, 0].item() != 0.0
    assert pred.grad[0, 0, 1, 1].item() == 0.0
