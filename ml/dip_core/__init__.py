"""Deep Image Prior Core Package for Thermal Image Restoration."""

import sys
import types

# Ensure torch compile hooks don't fail if host environment has broken sympy/mpmath
try:
    import torch._dynamo
except Exception:
    if "torch._dynamo" not in sys.modules:
        m = types.ModuleType("torch._dynamo")
        m.disable = lambda fn=None, recursive=True: (lambda f: f) if fn is None else fn
        m.is_compiling = lambda: False
        m.graph_break = lambda: None
        m.mark_step = lambda: None
        sys.modules["torch._dynamo"] = m

from dip_core.config import (
    DipTrainingConfig,
    ModelConfig,
    DamageConfig,
    RestorationTask,
    get_device,
)
from dip_core.models.unet_generator import UNetGenerator, UNetDIP
from dip_core.models.conv_baseline_generator import ConvBaselineGenerator, ConvDIPBaseline
from dip_core.pipeline.restoration_pipeline import run_restoration, RestorationOutput
from dip_core.pipeline.dip_trainer import DipTrainer, TrainingResult
from dip_core.metrics import calculate_psnr, calculate_ssim, calculate_masked_psnr
from dip_core.baselines import evaluate_baselines

__all__ = [
    "DipTrainingConfig",
    "ModelConfig",
    "DamageConfig",
    "RestorationTask",
    "get_device",
    "UNetGenerator",
    "UNetDIP",
    "ConvBaselineGenerator",
    "ConvDIPBaseline",
    "run_restoration",
    "RestorationOutput",
    "DipTrainer",
    "TrainingResult",
    "calculate_psnr",
    "calculate_ssim",
    "calculate_masked_psnr",
    "evaluate_baselines",
]
