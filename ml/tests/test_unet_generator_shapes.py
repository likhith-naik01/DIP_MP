"""
Tests verifying tensor shapes through all 5 levels of the U-Net generator.
"""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent.parent
ml_dir = repo_root / "ml"
for p in [repo_root, ml_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import unittest
import torch
from dip_core.models.unet_generator import UNetGenerator
from dip_core.models.conv_baseline_generator import ConvBaselineGenerator

class TestUNetShapes(unittest.TestCase):
    def test_unet_output_shape_matches_input(self):
        """Verify (1, 32, H, W) noise tensor produces (1, 1, H, W) output."""
        model = UNetGenerator(in_channels=32, out_channels=1)
        z = torch.randn(1, 32, 128, 128)
        out = model(z)
        self.assertEqual(out.shape, (1, 1, 128, 128))
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))

    def test_unet_various_divisible_dimensions(self):
        """Verify divisible dimensions (64, 96, 160) work properly."""
        model = UNetGenerator(in_channels=32, out_channels=1)
        for dim in [64, 96, 160]:
            z = torch.randn(1, 32, dim, dim)
            out = model(z)
            self.assertEqual(out.shape, (1, 1, dim, dim))

    def test_conv_baseline_output_shape(self):
        """Verify baseline generator produces exact output shape."""
        model = ConvBaselineGenerator(in_channels=32, out_channels=1)
        z = torch.randn(1, 32, 64, 64)
        out = model(z)
        self.assertEqual(out.shape, (1, 1, 64, 64))

if __name__ == "__main__":
    unittest.main()
