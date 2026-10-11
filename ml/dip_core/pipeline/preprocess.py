import os
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


def get_device(requested: Optional[str] = None) -> torch.device:
    """Return torch.device: cuda if available and requested/default, else cpu."""
    if requested and requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_image(
    image_input: Union[str, Image.Image, np.ndarray],
    grayscale: bool = True,
) -> np.ndarray:
    """
    Load an image from a filepath, PIL Image, or numpy array.
    Converts to 1-channel grayscale and normalizes pixel values to [0, 1].

    Returns:
        np.ndarray: float32 array of shape (H, W) in range [0, 1].
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at path: {image_input}")
        pil_img = Image.open(image_input)
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
    return np.clip(arr, 0.0, 1.0)


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int = 32,
    mode: str = "reflect",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pad tensor (B, C, H, W) so that H and W are divisible by `multiple`.
    Keeps the original size (H, W) for restoring later.
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


def unpad_tensor(
    tensor: torch.Tensor,
    original_size: Tuple[int, int],
) -> torch.Tensor:
    """Crop padded tensor back to its original (H, W) resolution."""
    orig_h, orig_w = original_size
    curr_h, curr_w = tensor.shape[-2], tensor.shape[-1]

    if curr_h == orig_h and curr_w == orig_w:
        return tensor

    pad_h = curr_h - orig_h
    pad_w = curr_w - orig_w

    pad_left = pad_w // 2
    pad_top = pad_h // 2

    return tensor[..., pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]


def to_torch_tensor(
    img_np: np.ndarray,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Convert numpy array (H, W) to (1, 1, H, W) float32 torch tensor on target device."""
    dev = get_device(device if isinstance(device, str) else None) if (device is None or isinstance(device, str)) else device
    if img_np.ndim == 2:
        tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    elif img_np.ndim == 3:
        tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected array shape: {img_np.shape}")
    return tensor.float().to(dev)


def build_mask(
    mask_input: Optional[Union[str, Image.Image, np.ndarray, torch.Tensor]] = None,
    target_shape: Optional[Tuple[int, int]] = None,
    image_for_autodetect: Optional[np.ndarray] = None,
    task: str = "denoise",
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Build binary mask tensor M of shape (1, 1, H, W).
    Convention: valid pixel = 1, missing / corrupted pixel = 0.

    Modes:
      1. task == 'denoise': mask is all ones.
      2. User-provided mask: parsed from image/array (0 = corrupted, 1 = valid).
      3. Auto-detected: detects dead pixels (exact 0.0 or <= 1e-4) or sensor saturation (>= 0.999).
    """
    dev = get_device(device if isinstance(device, str) else None) if (device is None or isinstance(device, str)) else device

    if task == "denoise" and mask_input is None and image_for_autodetect is None:
        if target_shape is None:
            raise ValueError("target_shape required when creating default all-ones mask.")
        return torch.ones((1, 1, target_shape[0], target_shape[1]), dtype=torch.float32, device=dev)

    # 1. User provided mask
    if mask_input is not None:
        if isinstance(mask_input, torch.Tensor):
            mask_t = mask_input.float().to(dev)
            if mask_t.ndim == 2:
                mask_t = mask_t.unsqueeze(0).unsqueeze(0)
            elif mask_t.ndim == 3:
                mask_t = mask_t.unsqueeze(0)
            return (mask_t > 0.5).float()

        mask_np = load_image(mask_input, grayscale=True)
        if target_shape and mask_np.shape != target_shape:
            mask_np = cv2.resize(mask_np, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_t = to_torch_tensor(mask_np, device=dev)
        return (mask_t > 0.5).float()

    # 2. Auto-detection from image (dead / stuck pixels or saturation)
    if image_for_autodetect is not None:
        h, w = image_for_autodetect.shape[:2]
        # Valid pixels = not dead (value > 0.001) and not saturated (value < 0.999)
        valid = (image_for_autodetect > 0.005) & (image_for_autodetect < 0.995)
        mask_np = valid.astype(np.float32)
        return to_torch_tensor(mask_np, device=dev)

    if target_shape:
        return torch.ones((1, 1, target_shape[0], target_shape[1]), dtype=torch.float32, device=dev)
    raise ValueError("Cannot build mask without either mask_input, image_for_autodetect, or target_shape.")


def simulate_damage(
    image: np.ndarray,
    noise_type: str = "gaussian",
    noise_std: float = 0.1,
    snr_db: Optional[float] = None,
    pixel_dropout_rate: float = 0.05,
    add_holes: bool = False,
    hole_type: str = "rect",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate artificial damage for thermal testing and benchmarking.

    Args:
        image: Clean float32 image in [0, 1].
        noise_type: 'gaussian' or 'salt_and_pepper'.
        noise_std: Standard deviation of Gaussian noise.
        snr_db: Optional signal-to-noise ratio in dB (overrides noise_std).
        pixel_dropout_rate: Fraction of random pixels to drop (missing).
        add_holes: Whether to cut out missing rectangular or line holes.
        hole_type: 'rect' or 'lines' (scratches).

    Returns:
        Tuple of (degraded_image, binary_mask) where mask has 1=valid, 0=missing.
    """
    h, w = image.shape[:2]
    degraded = image.copy()
    mask = np.ones((h, w), dtype=np.float32)

    # 1. Additive Noise
    if snr_db is not None:
        signal_power = np.mean(image**2)
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise_std = float(np.sqrt(noise_power))

    if noise_std > 0:
        noise = np.random.normal(0, noise_std, (h, w)).astype(np.float32)
        degraded = np.clip(degraded + noise, 0.0, 1.0)

    # 2. Random Pixel Dropout (Dead pixels / salt-and-pepper)
    if pixel_dropout_rate > 0:
        drop_mask = np.random.uniform(0, 1, (h, w)) < pixel_dropout_rate
        degraded[drop_mask] = 0.0
        mask[drop_mask] = 0.0

    # 3. Structural Holes
    if add_holes:
        if hole_type == "rect":
            # Cut out 1 or 2 rectangular missing regions (e.g. sensor block failure)
            num_holes = np.random.randint(1, 3)
            for _ in range(num_holes):
                bh = int(h * np.random.uniform(0.15, 0.25))
                bw = int(w * np.random.uniform(0.15, 0.25))
                y0 = np.random.randint(0, h - bh)
                x0 = np.random.randint(0, w - bw)
                degraded[y0 : y0 + bh, x0 : x0 + bw] = 0.0
                mask[y0 : y0 + bh, x0 : x0 + bw] = 0.0
        elif hole_type == "lines":
            # Cut out line scratches
            for _ in range(6):
                y1, x1 = np.random.randint(0, h), np.random.randint(0, w)
                y2, x2 = np.random.randint(0, h), np.random.randint(0, w)
                cv2.line(mask, (x1, y1), (x2, y2), 0.0, thickness=np.random.randint(2, 4))
            degraded = degraded * mask

    return degraded.astype(np.float32), mask.astype(np.float32)
