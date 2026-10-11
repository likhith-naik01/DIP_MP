"""Unit tests verifying mathematical behavior of Masked MSE loss."""
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent.parent
ml_dir = repo_root / "ml"
for p in [repo_root, ml_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import unittest
import torch
from dip_core.pipeline.masked_loss import MaskedMSELoss

class TestMaskedLoss(unittest.TestCase):
    def test_masked_loss_ignores_masked_regions(self):
        """Verify changing values in masked regions (M=0) has zero effect on loss."""
        criterion = MaskedMSELoss()

        pred = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
        target1 = torch.tensor([[[[0.5, 0.5], [0.5, 0.0]]]])
        target2 = torch.tensor([[[[0.5, 0.5], [0.5, 999.0]]]])  # Wildly different at [1, 1]
        mask = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]])      # Missing at [1, 1]

        loss1 = criterion(pred, target1, mask)
        loss2 = criterion(pred, target2, mask)

        self.assertAlmostEqual(loss1.item(), loss2.item(), places=5)
        self.assertAlmostEqual(loss1.item(), 0.0, places=5)

    def test_masked_loss_gradient_zero_on_missing_pixels(self):
        """Verify backpropagation gradients are strictly zero on missing pixels."""
        criterion = MaskedMSELoss()

        pred = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], requires_grad=True)
        target = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
        mask = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]])  # [1, 1] is masked out

        loss = criterion(pred, target, mask)
        loss.backward()

        self.assertNotEqual(pred.grad[0, 0, 0, 0].item(), 0.0)
        self.assertEqual(pred.grad[0, 0, 1, 1].item(), 0.0)

if __name__ == "__main__":
    unittest.main()
