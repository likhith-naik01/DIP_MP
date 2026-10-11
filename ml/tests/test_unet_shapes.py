try:
    import pytest
except ImportError:
    pass
import torch
import sys
import os

# Add ml root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dip_core.models.unet import UNetDIP
from dip_core.models.conv_baseline import ConvDIPBaseline


def test_unet_output_shape():
    """Verify U-Net forward pass preserves exact spatial dimensions and generates 1-channel output."""
    model = UNetDIP(in_channels=32, out_channels=1, skip_channels=4)
    model.eval()

    # Square shape divisible by 32
    x = torch.randn(1, 32, 128, 128)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 1, 128, 128), f"Expected (1, 1, 128, 128), got {out.shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0, "Output must be normalized within [0, 1] due to sigmoid"


def test_unet_rectangular_shape():
    """Verify U-Net handles non-square spatial resolutions divisible by 32."""
    model = UNetDIP(in_channels=32, out_channels=1, skip_channels=4)
    model.eval()

    x = torch.randn(1, 32, 96, 160)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 1, 96, 160), f"Expected (1, 1, 96, 160), got {out.shape}"


def test_conv_baseline_output_shape():
    """Verify ConvDIPBaseline generates expected output shape without skip connections."""
    model = ConvDIPBaseline(in_channels=32, out_channels=1)
    model.eval()

    x = torch.randn(1, 32, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 1, 64, 64), f"Expected (1, 1, 64, 64), got {out.shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0
