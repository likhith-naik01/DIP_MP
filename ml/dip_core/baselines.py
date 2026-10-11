from typing import Dict, Any, Optional
import numpy as np
import cv2
from dip_core.metrics import calculate_psnr, calculate_ssim


def apply_mean_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Mean / Box blur filter (e.g. 3x3 or 7x7)."""
    img_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    blurred = cv2.blur(img_u8, (ksize, ksize))
    return (blurred.astype(np.float32) / 255.0)


def apply_median_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median blur filter."""
    img_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.medianBlur(img_u8, ksize)
    return (blurred.astype(np.float32) / 255.0)


def apply_gaussian_filter(image: np.ndarray, ksize: int = 5, sigma: float = 1.2) -> np.ndarray:
    """Gaussian blur filter."""
    img_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(img_u8, (ksize, ksize), sigma)
    return (blurred.astype(np.float32) / 255.0)


def apply_inpaint_baseline(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "telea",
    inpaint_radius: int = 5,
) -> np.ndarray:
    """OpenCV classical inpainting (telea or ns)."""
    img_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    # Mask for cv2.inpaint: non-zero pixels are the area to be inpainted (missing = 255)
    cv_mask = ((mask < 0.5) * 255).astype(np.uint8)
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    inpainted = cv2.inpaint(img_u8, cv_mask, inpaint_radius, flag)
    return (inpainted.astype(np.float32) / 255.0)


def evaluate_baselines(
    degraded: np.ndarray,
    mask: Optional[np.ndarray] = None,
    reference: Optional[np.ndarray] = None,
    task: str = "denoise",
) -> Dict[str, Dict[str, Any]]:
    """
    Run mean 3x3, mean 7x7, median, and gaussian filters.
    Computes PSNR and SSIM against clean reference if available.
    """
    results: Dict[str, Dict[str, Any]] = {}

    mean_3 = apply_mean_filter(degraded, ksize=3)
    mean_7 = apply_mean_filter(degraded, ksize=7)
    med_5 = apply_median_filter(degraded, ksize=5)
    gauss_5 = apply_gaussian_filter(degraded, ksize=5, sigma=1.2)

    filters = [
        ("mean_3x3", mean_3),
        ("mean_7x7", mean_7),
        ("median", med_5),
        ("gaussian", gauss_5),
    ]

    for name, img in filters:
        entry: Dict[str, Any] = {"image": img}
        if reference is not None:
            entry["psnr"] = calculate_psnr(img, reference, mask=mask)
            entry["ssim"] = calculate_ssim(img, reference, mask=mask)
        results[name] = entry

    if task == "inpaint" and mask is not None:
        telea = apply_inpaint_baseline(degraded, mask, method="telea")
        entry_telea: Dict[str, Any] = {"image": telea}
        if reference is not None:
            entry_telea["psnr"] = calculate_psnr(telea, reference)
            entry_telea["ssim"] = calculate_ssim(telea, reference)
        results["opencv_telea"] = entry_telea

    return results
