"""Compatibility wrapper redirecting to image_preprocessing.py."""

from dip_core.pipeline.image_preprocessing import (
    load_image,
    pad_to_multiple_of_32 as pad_to_multiple,
    crop_back_to_original_size as unpad_tensor,
    to_tensor as to_torch_tensor,
    build_valid_mask as build_mask,
    get_device,
)

__all__ = [
    "load_image",
    "pad_to_multiple",
    "unpad_tensor",
    "to_torch_tensor",
    "build_mask",
    "get_device",
]
