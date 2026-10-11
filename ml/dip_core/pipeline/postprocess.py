import os
from typing import Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
import cv2
from dip_core.pipeline.preprocess import unpad_tensor


def tensor_to_numpy(
    tensor: torch.Tensor,
    original_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Convert PyTorch tensor (1, C, H, W) to float32 numpy array (H, W) or (H, W, C) in [0, 1].
    Unpads back to original size if specified.
    """
    if original_size is not None:
        tensor = unpad_tensor(tensor, original_size)

    arr = tensor.detach().cpu().squeeze().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        # Convert (C, H, W) to (H, W, C)
        arr = arr.transpose((1, 2, 0))
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    return arr


def apply_thermal_colormap(image: np.ndarray, colormap: str = "ironbow") -> np.ndarray:
    """
    Apply pseudo-color mapping for thermal imagery:
      - 'grayscale': standard monochrome infrared
      - 'ironbow': classic FLIR thermal palette (purple -> orange -> yellow)
      - 'inferno': matplotlib inferno
      - 'jet': blue -> green -> red
    Returns RGB uint8 numpy array (H, W, 3).
    """
    u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)

    if colormap == "grayscale":
        return np.stack([u8, u8, u8], axis=-1)
    elif colormap == "jet":
        rgb = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    elif colormap == "inferno":
        rgb = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    elif colormap in ("ironbow", "flir"):
        # Flir/Ironbow approximation using COLORMAP_COLORCUBE / HOT / MAGMA
        rgb = cv2.applyColorMap(u8, cv2.COLORMAP_MAGMA)
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    else:
        return np.stack([u8, u8, u8], axis=-1)


def save_image(
    image: Union[np.ndarray, torch.Tensor],
    output_path: str,
    original_size: Optional[Tuple[int, int]] = None,
    colormap: Optional[str] = None,
) -> str:
    """Save float32 [0, 1] numpy array or tensor to PNG image file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if isinstance(image, torch.Tensor):
        arr = tensor_to_numpy(image, original_size)
    else:
        arr = image

    if colormap and colormap != "grayscale":
        rgb_arr = apply_thermal_colormap(arr, colormap)
        pil_img = Image.fromarray(rgb_arr)
    else:
        u8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(u8)

    pil_img.save(output_path, format="PNG")
    return output_path
