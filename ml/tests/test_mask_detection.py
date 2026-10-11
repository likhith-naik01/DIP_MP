"""
Tests for automatic mask detection algorithms in thermal images.
"""
import numpy as np
import pytest
from dip_core.pipeline.mask_detection import (
    detect_dead_pixels,
    detect_saturation,
    detect_scratch_lines,
    auto_detect_damage_mask,
)

def test_detect_dead_pixels_recall():
    """Verify injected dead pixels (0s) are detected with recall > 0.9."""
    rng = np.random.RandomState(42)
    # Generate realistic thermal background in range [0.3, 0.7]
    base = rng.uniform(0.3, 0.7, size=(100, 100)).astype(np.float32)
    
    # Inject isolated dead pixels
    dead_coords = [(15, 20), (35, 45), (60, 80), (80, 15), (90, 70)]
    corrupted = base.copy()
    ground_truth_mask = np.zeros((100, 100), dtype=np.float32)
    for r, c in dead_coords:
        corrupted[r, c] = 0.0
        ground_truth_mask[r, c] = 1.0
        
    detected_mask = detect_dead_pixels(corrupted, dead_val=0.01)
    
    # Check that detected mask identifies the dead pixels
    # Note: mask convention in detect_dead_pixels returns 1 for damage
    detected_count = 0
    for r, c in dead_coords:
        if detected_mask[r, c] > 0.5:
            detected_count += 1
            
    recall = detected_count / len(dead_coords)
    assert recall >= 0.9, f"Expected recall >= 0.9, got {recall}"

def test_detect_saturation():
    """Verify saturated pixels (>=0.99) are detected."""
    img = np.full((50, 50), 0.5, dtype=np.float32)
    img[10:15, 10:15] = 1.0  # Saturated patch
    
    mask = detect_saturation(img, sat_val=0.99)
    assert np.all(mask[10:15, 10:15] == 1.0)
    assert np.all(mask[0:5, 0:5] == 0.0)

def test_detect_scratch_lines():
    """Verify thin scratch line is detected."""
    img = np.full((60, 60), 0.5, dtype=np.float32)
    img[30, 10:50] = 0.0  # Horizontal scratch
    
    mask = detect_scratch_lines(img, length_threshold=15)
    assert np.sum(mask[30, 10:50]) > 0

def test_auto_detect_damage_mask_valid_format():
    """Verify auto_detect_damage_mask returns valid binary array."""
    img = np.random.uniform(0.1, 0.9, size=(64, 64)).astype(np.float32)
    mask = auto_detect_damage_mask(img)
    assert mask.shape == (64, 64)
    assert mask.dtype == np.float32
    assert set(np.unique(mask)).issubset({0.0, 1.0})
