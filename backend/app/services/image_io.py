"""
Image input/output helpers: Base64 encoding/decoding, format normalization, and validation.
"""
import io
import base64
from typing import Tuple, Optional
import numpy as np
from PIL import Image
from dip_core.pipeline.image_postprocessing import apply_thermal_colormap
from backend.app.config import MAX_IMAGE_DIM

def load_image_from_bytes(file_bytes: bytes) -> np.float32:
    """
    Reads image bytes, validates dimensions (downscales if > MAX_IMAGE_DIM),
    converts to grayscale float32 [0.0, 1.0].
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / float(max(w, h))
        new_w = max(32, int(w * scale))
        new_h = max(32, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)

def load_mask_from_bytes(file_bytes: bytes, target_shape: Tuple[int, int]) -> np.float32:
    """
    Loads user-provided or drawn mask from bytes and aligns it to target shape.
    Output: 1.0 for valid, 0.0 for damaged.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    if (img.height, img.width) != target_shape:
        img = img.resize((target_shape[1], target_shape[0]), Image.Resampling.NEAREST)
        
    arr = np.array(img, dtype=np.float32) / 255.0
    # Standardize binary mask: pixels >= 0.5 are valid (1.0), else 0.0
    mask = (arr >= 0.5).astype(np.float32)
    return mask

def array_to_base64_png(arr: np.ndarray, colormap: str = "grayscale") -> str:
    """
    Converts 2D float32 numpy array [0, 1] into a base64 encoded PNG data URI.
    """
    if colormap == "grayscale":
        uint8_arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        img = Image.fromarray(uint8_arr, mode="L")
    else:
        # Thermal colormap (ironbow, inferno, viridis)
        rgb_arr = apply_thermal_colormap(arr, colormap=colormap)
        img = Image.fromarray(rgb_arr, mode="RGB")
        
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

def array_to_png_bytes(arr: np.ndarray, colormap: str = "grayscale") -> bytes:
    """
    Converts array to raw PNG bytes for file download.
    """
    if colormap == "grayscale":
        uint8_arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        img = Image.fromarray(uint8_arr, mode="L")
    else:
        rgb_arr = apply_thermal_colormap(arr, colormap=colormap)
        img = Image.fromarray(rgb_arr, mode="RGB")
        
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return buffered.getvalue()
