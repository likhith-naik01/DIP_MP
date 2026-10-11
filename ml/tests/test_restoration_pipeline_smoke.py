"""
Smoke test for end-to-end restoration pipeline.
Ensures integration between preprocessor, UNet generator, loss, trainer and postprocessor.
"""
import numpy as np
import pytest
from dip_core.config import DipTrainingConfig, RestorationTask
from dip_core.pipeline.restoration_pipeline import run_restoration

def test_restoration_pipeline_smoke_run():
    """Verify run_restoration executes 5 iterations without runtime errors."""
    # Synthetic small 64x64 test image
    img = np.full((64, 64), 0.5, dtype=np.float32)
    # Add simple gradient
    for i in range(64):
        img[i, :] += (i / 64.0) * 0.4
    
    # Mask with central 8x8 hole
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
        task=RestorationTask.INPAINTING,
        model_name="unet",
        training_config=cfg
    )
    
    assert output is not None
    assert output.restored_array.shape == (64, 64)
    assert len(output.losses) == 5
    assert output.metrics is not None
    assert "psnr" in output.metrics
    assert "ssim" in output.metrics
    assert output.runtime_seconds >= 0.0
