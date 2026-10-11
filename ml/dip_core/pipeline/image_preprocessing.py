"""Image loading, reflection padding, device transfers, and mask preparation."""

import io
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

from dip_core.config import get_device


def load_image(
    image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
    grayscale: bool = True,
) -> np.ndarray:
    """
    Load an image from filepath, byte buffer, PIL Image, or numpy array.
    Normalizes to float32 in range [0.0, 1.0].
    Thermal frames are represented as single-channel grayscale (H, W).
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image not found at path: {image_input}")
        pil_img = Image.open(p)
    elif isinstance(image_input, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, np.ndarray):
        arr = image_input.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if grayscale and arr.ndim == 3:
            arr = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
        return np.clip(arr, 0.0, 1.0).astype(np.float32)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    if grayscale:
        pil_img = pil_img.convert("L")
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def pad_to_multiple_of_32(
    tensor: torch.Tensor,
    multiple: int = 32,
    mode: str = "reflect",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pad tensor (B, C, H, W) so H and W are divisible by `multiple` (32).
    Returns padded tensor and original (H, W) for post-processing restoration.
    """
    h, w = tensor.shape[-2], tensor.shape[-1]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    if pad_h == 0 and pad_w == 0:
        return tensor, (h, w)

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    try:
        padded = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)
    except Exception:
        padded = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

    return padded, (h, w)


def crop_back_to_original_size(
    tensor: torch.Tensor,
    original_size: Tuple[int, int],
) -> torch.Tensor:
    """Crop padded tensor back to its exact original (H, W) spatial dimensions."""
    orig_h, orig_w = original_size
    curr_h, curr_w = tensor.shape[-2], tensor.shape[-1]

    if curr_h == orig_h and curr_w == orig_w:
        return tensor

    pad_left = (curr_w - orig_w) // 2
    pad_top = (curr_h - orig_h) // 2

    return tensor[..., pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]


def to_tensor(
    img_np: np.ndarray,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Convert numpy array (H, W) to (1, 1, H, W) float32 torch tensor."""
    dev = get_device(device if isinstance(device, str) else None) if (device is None or isinstance(device, str)) else device
    if img_np.ndim == 2:
        tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    elif img_np.ndim == 3:
        tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected array shape: {img_np.shape}")
    return tensor.float().to(dev)


def to_device(tensor: torch.Tensor, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """Move tensor to target device."""
    dev = get_device(device if isinstance(device, str) else None) if (device is None or isinstance(device, str)) else device
    return tensor.to(dev)


def build_valid_mask(
    mask_input: Optional[Union[str, Path, bytes, Image.Image, np.ndarray, torch.Tensor]] = None,
    target_shape: Optional[Tuple[int, int]] = None,
    image_for_autodetect: Optional[np.ndarray] = None,
    task: str = "denoise",
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Build binary valid mask M of shape (1, 1, H, W).
    Convention: 1.0 = valid pixel, 0.0 = missing / damaged pixel.

    Rules:
      - For task == 'denoise': M is all ones.
      - User masks: standard convention has white pixels = damaged/selected region.
        Invert so damaged pixels = 0, valid pixels = 1.
      - Auto-detected: delegates to mask_detection module.
    """
    dev = get_device(device if isinstance(device, str) else None) if (device is None or isinstance(device, str)) else device

    if task == "denoise" and mask_input is None and image_for_autodetect is None:
        if target_shape is None:
            raise ValueError("target_shape required when generating all-ones mask.")
        return torch.ones((1, 1, target_shape[0], target_shape[1]), dtype=torch.float32, device=dev)

    # 1. User-provided mask
    if mask_input is not None:
        if isinstance(mask_input, torch.Tensor):
            mask_t = mask_input.float().to(dev)
            if mask_t.ndim == 2:
                mask_t = mask_t.unsqueeze(0).unsqueeze(0)
            elif mask_t.ndim == 3:
                mask_t = mask_t.unsqueeze(0)
            # If user drew white (1) as damaged, invert so valid = 1, damaged = 0
            # If mask already has mostly 1s, keep as is
            if mask_t.mean() < 0.5:
                # White was drawn on black background -> invert
                mask_t = 1.0 - (mask_t > 0.5).float()
            else:
                mask_t = (mask_t > 0.5).float()
            return mask_t

        mask_np = load_image(mask_input, grayscale=True)
        if target_shape and mask_np.shape != target_shape:
            mask_np = cv2.resize(mask_np, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

        # Invert if white indicates damage
        if mask_np.mean() < 0.5:
            mask_np = 1.0 - (mask_np > 0.5).astype(np.float32)
        else:
            mask_np = (mask_np > 0.5).astype(np.float32)

        return to_tensor(mask_np, device=dev)

    # 2. Auto-detection from image
    if image_for_autodetect is not None:
        from dip_core.pipeline.mask_detection import detect_damage_mask
        mask_np, _ = detect_damage_mask(image_for_autodetect)
        return to_tensor(mask_np, device=dev)

    if target_shape:
        return torch.ones((1, 1, target_shape[0], target_shape[1]), dtype=torch.float32, device=dev)
    raise ValueError("Cannot build mask without either mask_input, image_for_autodetect, or target_shape.")
