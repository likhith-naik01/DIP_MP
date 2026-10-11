"""Compatibility wrapper redirecting to image_postprocessing.py."""

from dip_core.pipeline.image_postprocessing import (
    composite_restored_image,
    apply_thermal_colormap,
    save_image,
    to_uint8,
)

__all__ = ["composite_restored_image", "apply_thermal_colormap", "save_image", "to_uint8"]
