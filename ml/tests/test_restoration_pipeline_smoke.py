"""
Smoke test for end-to-end restoration pipeline.
Ensures integration between preprocessor, UNet generator, loss, trainer and postprocessor.
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
from dip_core.config import DipTrainingConfig, RestorationTask
from dip_core.pipeline.restoration_pipeline import run_restoration

class TestRestorationPipelineSmoke(unittest.TestCase):
    def test_restoration_pipeline_smoke_run(self):
        """Verify run_restoration executes 5 iterations without runtime errors."""
        img = np.full((64, 64), 0.5, dtype=np.float32)
        for i in range(64):
            img[i, :] += (i / 64.0) * 0.4
        
        mask = np.ones((64, 64), dtype=np.float32)
        mask[28:36, 28:36] = 0.0
        
        cfg = DipTrainingConfig(
            iterations=5,
            save_every=2,
            log_every=1,
            learning_rate=0.01,
            device="cpu"
        )
        
        output = run_restoration(
            image_input=img,
            mask_input=mask,
            task=RestorationTask.INPAINT,
            model_name="unet",
            training_config=cfg,
            reference_image=img
        )
        
        self.assertIsNotNone(output)
        self.assertEqual(output.restored_array.shape, (64, 64))
        self.assertEqual(len(output.losses), 5)
        self.assertIsNotNone(output.metrics)
        self.assertIn("psnr", output.metrics)
        self.assertIn("ssim", output.metrics)
        self.assertGreaterEqual(output.runtime_seconds, 0.0)

if __name__ == "__main__":
    unittest.main()
