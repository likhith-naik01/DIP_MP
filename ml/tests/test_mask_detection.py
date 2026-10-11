"""
Tests for automatic mask detection algorithms in thermal images.
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
from dip_core.pipeline.mask_detection import (
    detect_dead_pixels,
    detect_saturation,
    detect_scratch_lines,
    auto_detect_damage_mask,
)

class TestMaskDetection(unittest.TestCase):
    def test_detect_dead_pixels_recall(self):
        """Verify injected dead pixels (0s) are detected with recall > 0.9."""
        rng = np.random.RandomState(42)
        base = rng.uniform(0.3, 0.7, size=(100, 100)).astype(np.float32)
        
        dead_coords = [(15, 20), (35, 45), (60, 80), (80, 15), (90, 70)]
        corrupted = base.copy()
        for r, c in dead_coords:
            corrupted[r, c] = 0.0
            
        detected_mask = detect_dead_pixels(corrupted, dead_val=0.01)
        detected_count = sum(1 for r, c in dead_coords if detected_mask[r, c] > 0.5)
        recall = detected_count / len(dead_coords)
        self.assertGreaterEqual(recall, 0.9)

    def test_detect_saturation(self):
        """Verify saturated pixels (>=0.99) are detected."""
        img = np.full((50, 50), 0.5, dtype=np.float32)
        img[10:15, 10:15] = 1.0
        mask = detect_saturation(img, sat_val=0.99)
        self.assertTrue(np.all(mask[10:15, 10:15] == 1.0))
        self.assertTrue(np.all(mask[0:5, 0:5] == 0.0))

    def test_detect_scratch_lines(self):
        """Verify thin scratch line is detected."""
        img = np.full((60, 60), 0.5, dtype=np.float32)
        img[30, 10:50] = 0.0
        mask = detect_scratch_lines(img, length_threshold=15)
        self.assertGreater(np.sum(mask[30, 10:50]), 0)

    def test_auto_detect_damage_mask_valid_format(self):
        """Verify auto_detect_damage_mask returns valid binary array."""
        img = np.random.uniform(0.1, 0.9, size=(64, 64)).astype(np.float32)
        mask = auto_detect_damage_mask(img)
        self.assertEqual(mask.shape, (64, 64))
        self.assertEqual(mask.dtype, np.float32)
        self.assertTrue(set(np.unique(mask)).issubset({0.0, 1.0}))

if __name__ == "__main__":
    unittest.main()
