"""Postprocessing: unpadding, compositing/blending, colormaps, and disk saving."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2


def composite_restored_image(
    degraded: np.ndarray,
    restored: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Blend restored output with original valid pixels:
      composite = mask * degraded + (1 - mask) * restored
    where mask has 1.0 for valid pixels, 0.0 for damaged/missing pixels.
    If all pixels in mask are valid (e.g. pure denoising/enhancement), returns restored array.
    """
    m = np.clip(mask, 0.0, 1.0)
    if np.all(m >= 0.999):
        return np.clip(restored, 0.0, 1.0).astype(np.float32)
    composite = (m * degraded) + ((1.0 - m) * restored)
    return np.clip(composite, 0.0, 1.0).astype(np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float32 image [0.0, 1.0] to uint8 [0, 255]."""
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def tensor_to_numpy(tensor, orig_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Convert torch.Tensor (1, 1, H, W) or (H, W) to 2D numpy array [0.0, 1.0]."""
    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().squeeze().numpy()
    else:
        arr = np.asarray(tensor)
    if orig_size is not None and len(orig_size) == 2 and arr.ndim >= 2:
        h, w = orig_size
        arr = arr[:h, :w]
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def apply_thermal_colormap(image: np.ndarray, colormap: str = "ironbow") -> np.ndarray:
    """
    Convert 1-channel thermal array to 3-channel RGB pseudo-color:
      - 'grayscale': standard monochrome infrared
      - 'ironbow': classic FLIR thermal infrared palette
      - 'inferno': inferno colormap
      - 'jet': blue-cyan-yellow-red rainbow
    """
    u8 = to_uint8(image)
    if colormap == "grayscale":
        return np.stack([u8, u8, u8], axis=-1)
    elif colormap == "jet":
        bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif colormap == "inferno":
        bgr = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif colormap in ("ironbow", "flir"):
        bgr = cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        return np.stack([u8, u8, u8], axis=-1)


def save_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    colormap: Optional[str] = None,
) -> Path:
    """Save float32 array as PNG image."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if colormap and colormap != "grayscale":
        rgb = apply_thermal_colormap(image, colormap)
        pil_img = Image.fromarray(rgb)
    else:
        u8 = to_uint8(image)
        pil_img = Image.fromarray(u8)

    pil_img.save(str(p), format="PNG")
    return p
