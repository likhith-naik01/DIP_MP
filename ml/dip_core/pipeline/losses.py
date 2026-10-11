"""Compatibility wrapper redirecting to masked_loss.py."""

from dip_core.pipeline.masked_loss import (
    MaskedMSELoss,
    masked_mse_loss,
    CompositeEdgeLoss,
    composite_edge_loss,
)

__all__ = [
    "MaskedMSELoss",
    "masked_mse_loss",
    "CompositeEdgeLoss",
    "composite_edge_loss",
]
