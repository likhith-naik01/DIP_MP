"""
Unit tests for the strong deblurring, sharpness metric, and detail restoration pipeline.
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
import cv2
from dip_core.pipeline.deblurring import (
    compute_sharpness_metrics,
    estimate_blur_psf,
    richardson_lucy_deconvolution,
    adaptive_unsharp_mask,
    strong_deblur_pipeline,
)
from dip_core.pipeline.restoration_pipeline import run_restoration
from dip_core.config import DipTrainingConfig, RestorationTask


class TestDeblurringPipeline(unittest.TestCase):
    def setUp(self):
        # Create a synthetic image with sharp edge transitions
        self.clean_img = np.zeros((64, 64), dtype=np.float32)
        self.clean_img[16:48, 16:48] = 0.8
        self.clean_img[24:40, 24:40] = 0.2
        # Add blur
        self.blurred_img = cv2.GaussianBlur(self.clean_img, (5, 5), 1.5)

    def test_sharpness_metrics_detect_blur(self):
        clean_metrics = compute_sharpness_metrics(self.clean_img)
        blurred_metrics = compute_sharpness_metrics(self.blurred_img)
        
        self.assertGreater(clean_metrics["laplacian_variance"], blurred_metrics["laplacian_variance"])
        self.assertGreater(clean_metrics["tenengrad_energy"], blurred_metrics["tenengrad_energy"])

    def test_strong_deblur_pipeline_increases_sharpness(self):
        restored, metrics = strong_deblur_pipeline(self.blurred_img, strength="strong")
        
        self.assertEqual(restored.shape, self.blurred_img.shape)
        self.assertGreater(metrics["sharpness_after"], metrics["sharpness_before"])
        self.assertGreater(metrics["sharpness_gain_pct"], 0.0)
        self.assertTrue(np.all(restored >= 0.0) and np.all(restored <= 1.0))

    def test_run_restoration_deblur_task(self):
        cfg = DipTrainingConfig(iterations=5, save_every=2, log_every=1, device="cpu")
        output = run_restoration(
            image=self.blurred_img,
            task="deblur",
            training_config=cfg
        )
        self.assertIsNotNone(output)
        self.assertEqual(output.restored_array.shape, (64, 64))
        self.assertIn("sharpness_gain_pct", output.metrics)
        self.assertGreater(output.metrics["sharpness_gain_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
