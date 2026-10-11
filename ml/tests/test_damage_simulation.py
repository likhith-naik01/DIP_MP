"""
Tests for damage simulation functions.
"""
import numpy as np
import pytest
from dip_core.pipeline.damage_simulation import (
    simulate_thermal_damage,
    add_gaussian_noise,
    add_dead_pixels,
    add_rectangular_holes,
    add_scratch_lines,
)
from dip_core.config import DamageConfig

def test_damage_simulation_reproducibility():
    """Verify same seed produces identical damage."""
    img = np.ones((80, 80), dtype=np.float32) * 0.5
    cfg = DamageConfig(
        snr_db=20.0,
        dead_pixel_ratio=0.02,
        rect_hole_count=2,
        scratch_line_count=2,
        seed=12345
    )
    
    corrupted_1, mask_1 = simulate_thermal_damage(img, cfg)
    corrupted_2, mask_2 = simulate_thermal_damage(img, cfg)
    
    np.testing.assert_allclose(corrupted_1, corrupted_2, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(mask_1, mask_2, rtol=1e-5, atol=1e-5)

def test_damage_simulation_different_seeds():
    """Verify different seeds produce different corrupted images."""
    img = np.ones((80, 80), dtype=np.float32) * 0.5
    cfg1 = DamageConfig(dead_pixel_ratio=0.05, seed=1)
    cfg2 = DamageConfig(dead_pixel_ratio=0.05, seed=2)
    
    corrupted_1, _ = simulate_thermal_damage(img, cfg1)
    corrupted_2, _ = simulate_thermal_damage(img, cfg2)
    
    assert not np.allclose(corrupted_1, corrupted_2)

def test_mask_is_binary_and_valid_coverage():
    """Verify mask has valid values in {0, 1} and valid damage region."""
    img = np.ones((64, 64), dtype=np.float32) * 0.6
    cfg = DamageConfig(
        dead_pixel_ratio=0.03,
        rect_hole_count=1,
        rect_hole_size=(10, 10),
        scratch_line_count=1,
        scratch_line_thickness=2,
        seed=42
    )
    corrupted, mask = simulate_thermal_damage(img, cfg)
    
    assert corrupted.shape == (64, 64)
    assert mask.shape == (64, 64)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    # Corrupted pixels should correspond to mask == 0
    # Mask convention: 1 = valid, 0 = damaged/masked out
    damaged_count = np.sum(mask == 0.0)
    assert damaged_count > 0, "Expected at least some damaged pixels"
