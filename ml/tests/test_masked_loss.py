try:
    import pytest
except ImportError:
    pass
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dip_core.pipeline.losses import MaskedMSELoss


def test_masked_loss_full_mask():
    """When mask is all ones, MaskedMSELoss matches standard MSE."""
    criterion = MaskedMSELoss()

    pred = torch.tensor([[[[0.5, 0.8], [0.2, 0.9]]]], requires_grad=True)
    target = torch.tensor([[[[0.0, 0.8], [0.2, 0.0]]]])
    mask = torch.ones_like(target)

    loss = criterion(pred, target, mask)
    # diff: [0.5, 0.0, 0.0, 0.9] -> diff^2: [0.25, 0.0, 0.0, 0.81]
    # sum = 1.06 / 4 = 0.265
    expected = 0.265
    assert abs(loss.item() - expected) < 1e-5


def test_masked_loss_ignores_missing_regions():
    """Corrupted pixels where M=0 must contribute 0 to the loss and gradient."""
    criterion = MaskedMSELoss()

    # Suppose bottom-right pixel is corrupted in target and has wild diff in prediction
    pred = torch.tensor([[[[1.0, 1.0], [1.0, 100.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 1.0], [1.0, -500.0]]]])
    # Mask is 0 at [1, 1]
    mask = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]])

    loss = criterion(pred, target, mask)
    # The first three pixels have diff=0, so valid sum is 0
    assert abs(loss.item()) < 1e-6

    loss.backward()
    # Gradient on missing region [1, 1] must be 0
    assert pred.grad[0, 0, 1, 1].item() == 0.0
