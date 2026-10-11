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

from dip_core.config import DIPConfig
from dip_core.models.unet import UNetDIP
from dip_core.models.conv_baseline import ConvDIPBaseline
from dip_core.pipeline.trainer import train_dip

__all__ = ["DIPConfig", "UNetDIP", "ConvDIPBaseline", "train_dip"]
