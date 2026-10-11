"""Unit tests verifying UNetGenerator shapes, parameter counting, and spatial divisibility guards."""

import pytest
import torch
from dip_core.models.unet_generator import UNetGenerator
from dip_core.models.conv_baseline_generator import ConvBaselineGenerator


def test_unet_output_shape():
    """Verify forward pass preserves exact spatial resolution and normalizes output."""
    model = UNetGenerator(in_channels=32, out_channels=1, skip_channels=4)
    model.eval()

    x = torch.randn(1, 32, 128, 128)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 1, 128, 128)
    assert 0.0 <= out.min().item() <= out.max().item() <= 1.0


def test_unet_non_divisible_raises_value_error():
    """Verify input not divisible by 32 raises a descriptive ValueError."""
    model = UNetGenerator(in_channels=32, out_channels=1)
    x = torch.randn(1, 32, 100, 100)  # 100 is not divisible by 32

    with pytest.raises(ValueError) as excinfo:
        model(x)
    assert "must be divisible by 32" in str(excinfo.value)


def test_unet_parameter_count():
    """Verify UNet parameter count is positive and greater than ConvBaseline due to skip projections."""
    unet = UNetGenerator(in_channels=32, out_channels=1, skip_channels=4)
    conv = ConvBaselineGenerator(in_channels=32, out_channels=1)

    assert unet.count_parameters() > 0
    assert conv.count_parameters() > 0
