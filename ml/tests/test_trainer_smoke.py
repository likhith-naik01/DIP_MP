import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dip_core.config import DIPConfig
from dip_core.pipeline.trainer import train_dip

class TestTrainerSmoke(unittest.TestCase):
    def test_trainer_smoke_20_iterations(self):
        """Smoke test running 20 iterations of trainer.py on a random image."""
        img = np.random.uniform(0.1, 0.9, (64, 64)).astype(np.float32)

        cfg = DIPConfig(
            iterations=20,
            learning_rate=0.01,
            log_interval=5,
            early_stopping=False,
            device="cpu",
        )

        logged_steps = []

        def callback(info):
            logged_steps.append(info["iteration"])

        result = train_dip(
            image_input=img,
            config=cfg,
            task="denoise",
            on_progress=callback,
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["restored_image"].shape, (64, 64))
        self.assertTrue(0.0 <= result["restored_image"].min() <= result["restored_image"].max() <= 1.0)
        self.assertEqual(len(result["loss_history"]), 20)
        self.assertEqual(result["metrics"]["total_iterations"], 20)
        self.assertGreater(len(logged_steps), 0)

    def test_trainer_smoke_inpaint(self):
        """Inpainting smoke test with synthetic mask."""
        img = np.random.uniform(0.1, 0.9, (64, 64)).astype(np.float32)
        mask = np.ones((64, 64), dtype=np.float32)
        mask[20:40, 20:40] = 0.0
        degraded = img * mask

        cfg = DIPConfig(
            iterations=20,
            learning_rate=0.01,
            log_interval=5,
            early_stopping=False,
            device="cpu",
        )

        result = train_dip(
            image_input=degraded,
            mask_input=mask,
            config=cfg,
            task="inpaint",
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["restored_image"].shape, (64, 64))
        self.assertIn("baselines", result)

    def test_trainer_cancellation(self):
        """Verify cancellation predicate halts training."""
        img = np.random.uniform(0.1, 0.9, (64, 64)).astype(np.float32)
        cfg = DIPConfig(iterations=100, device="cpu")

        def cancel_after_two():
            return True

        result = train_dip(
            image_input=img,
            config=cfg,
            is_cancelled=cancel_after_two,
        )

        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["iteration"], 1)

if __name__ == "__main__":
    unittest.main()
