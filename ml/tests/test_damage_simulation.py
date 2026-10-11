"""
Tests for damage simulation functions.
"""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent.parent
ml_dir = repo_root / "ml"
for p in [repo_root, ml_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import unittest
import numpy as np
from dip_core.pipeline.damage_simulation import (
    simulate_thermal_damage,
    add_gaussian_noise,
    add_dead_pixels,
    add_rectangular_holes,
    add_scratch_lines,
)
from dip_core.config import DamageConfig

class TestDamageSimulation(unittest.TestCase):
    def test_damage_simulation_reproducibility(self):
        """Verify same seed produces identical damage."""
        img = np.ones((80, 80), dtype=np.float32) * 0.5
        cfg = DamageConfig(
            dropout_ratio=0.02,
            num_rect_holes=2,
            num_line_holes=2,
            seed=12345
        )
        corrupted_1, mask_1 = simulate_thermal_damage(img, cfg)
        corrupted_2, mask_2 = simulate_thermal_damage(img, cfg)
        np.testing.assert_allclose(corrupted_1, corrupted_2, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(mask_1, mask_2, rtol=1e-5, atol=1e-5)

    def test_damage_simulation_different_seeds(self):
        """Verify different seeds produce different corrupted images."""
        img = np.ones((80, 80), dtype=np.float32) * 0.5
        cfg1 = DamageConfig(dropout_ratio=0.05, seed=1)
        cfg2 = DamageConfig(dropout_ratio=0.05, seed=2)
        corrupted_1, _ = simulate_thermal_damage(img, cfg1)
        corrupted_2, _ = simulate_thermal_damage(img, cfg2)
        self.assertFalse(np.allclose(corrupted_1, corrupted_2))

    def test_mask_is_binary_and_valid_coverage(self):
        """Verify mask has valid values in {0, 1} and non-zero damage region."""
        img = np.ones((64, 64), dtype=np.float32) * 0.6
        cfg = DamageConfig(
            dropout_ratio=0.03,
            num_rect_holes=1,
            num_line_holes=1,
            seed=42
        )
        corrupted, mask = simulate_thermal_damage(img, cfg)
        self.assertEqual(corrupted.shape, (64, 64))
        self.assertEqual(mask.shape, (64, 64))
        self.assertTrue(set(np.unique(mask)).issubset({0.0, 1.0}))
        damaged_count = np.sum(mask == 0.0)
        self.assertGreater(damaged_count, 0)

if __name__ == "__main__":
    unittest.main()
