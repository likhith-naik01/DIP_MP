"""Automated defect and damage mask detection for thermal sensor images."""

from typing import Tuple, Dict, Any
import numpy as np
import cv2


def detect_damage_mask(
    image: np.ndarray,
    dead_pixel_thresh: float = 1e-4,
    saturated_pixel_thresh: float = 0.999,
    median_ksize: int = 5,
    median_diff_thresh: float = 0.25,
    detect_scratches: bool = True,
    dilate_pixels: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Automatically detect damaged / corrupt pixels in a thermal image without user mask.

    Detects:
      1. Dead sensor pixels (value <= dead_pixel_thresh)
      2. Saturated sensor pixels (value >= saturated_pixel_thresh)
      3. Isolated impulse outliers (deviation from local median filter)
      4. Linear scratch defects via directional morphological kernels

    Returns:
      mask: np.ndarray of shape (H, W) with float32 values in {0.0, 1.0}
            (1.0 = valid pixel, 0.0 = damaged/missing pixel)
      stats: dictionary of defect detection statistics
    """
    img = np.clip(image, 0.0, 1.0).astype(np.float32)
    h, w = img.shape[:2]
    total_pixels = h * w

    # 1. Dead pixels
    dead_mask = img <= dead_pixel_thresh

    # 2. Saturated pixels
    sat_mask = img >= saturated_pixel_thresh

    # 3. Local median deviation (impulse / salt-and-pepper outliers)
    u8_img = (img * 255.0).astype(np.uint8)
    med_filtered = cv2.medianBlur(u8_img, median_ksize).astype(np.float32) / 255.0
    diff = np.abs(img - med_filtered)
    outlier_mask = diff > median_diff_thresh

    # 4. Scratch detection via morphological line kernels
    scratch_mask = np.zeros((h, w), dtype=bool)
    if detect_scratches:
        # Detect thin black lines (scratches)
        inv_u8 = 255 - u8_img
        for angle in [0, 45, 90, 135]:
            kernel_len = 9
            k = np.zeros((kernel_len, kernel_len), dtype=np.uint8)
            if angle == 0:
                k[kernel_len // 2, :] = 1
            elif angle == 90:
                k[:, kernel_len // 2] = 1
            elif angle == 45:
                np.fill_diagonal(k, 1)
            elif angle == 135:
                np.fill_diagonal(np.fliplr(k), 1)

            opened = cv2.morphologyEx(inv_u8, cv2.MORPH_OPEN, k)
            scratch_mask |= (opened > 80)

    # Combined damage mask (True = damaged)
    combined_damage = dead_mask | sat_mask | outlier_mask | scratch_mask

    # Dilate damaged regions by dilate_pixels
    if dilate_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate_pixels + 1, 2 * dilate_pixels + 1))
        dilated_u8 = cv2.dilate(combined_damage.astype(np.uint8), kernel)
        combined_damage = dilated_u8.astype(bool)

    # Valid mask convention: 1.0 = valid, 0.0 = damaged
    valid_mask = (~combined_damage).astype(np.float32)

    damaged_count = int(np.sum(combined_damage))
    stats = {
        "total_pixels": total_pixels,
        "damaged_pixels": damaged_count,
        "percentage_damaged": round((damaged_count / total_pixels) * 100.0, 2),
        "dead_pixels_detected": int(np.sum(dead_mask)),
        "saturated_pixels_detected": int(np.sum(sat_mask)),
        "outlier_pixels_detected": int(np.sum(outlier_mask)),
        "scratch_pixels_detected": int(np.sum(scratch_mask)),
        "method_used": "multi_cue_heuristic (dead + sat + median + morphology)",
    }

    return valid_mask, stats


def detect_dead_pixels(image: np.ndarray, dead_val: float = 0.01) -> np.ndarray:
    """Return binary mask: 1.0 for dead pixels (<= dead_val), 0.0 for others."""
    img = np.clip(image, 0.0, 1.0).astype(np.float32)
    return (img <= dead_val).astype(np.float32)


def detect_saturation(image: np.ndarray, sat_val: float = 0.99) -> np.ndarray:
    """Return binary mask: 1.0 for saturated pixels (>= sat_val), 0.0 for others."""
    img = np.clip(image, 0.0, 1.0).astype(np.float32)
    return (img >= sat_val).astype(np.float32)


def detect_scratch_lines(image: np.ndarray, length_threshold: int = 15) -> np.ndarray:
    """Return binary mask: 1.0 for scratch line defects, 0.0 for others."""
    img = np.clip(image, 0.0, 1.0).astype(np.float32)
    h, w = img.shape[:2]
    u8_img = (img * 255.0).astype(np.uint8)
    inv_u8 = 255 - u8_img
    scratch_mask = np.zeros((h, w), dtype=bool)
    
    k_len = max(3, length_threshold)
    k_h = np.zeros((k_len, k_len), dtype=np.uint8)
    k_h[k_len // 2, :] = 1
    opened_h = cv2.morphologyEx(inv_u8, cv2.MORPH_OPEN, k_h)
    scratch_mask |= (opened_h > 80)

    k_v = np.zeros((k_len, k_len), dtype=np.uint8)
    k_v[:, k_len // 2] = 1
    opened_v = cv2.morphologyEx(inv_u8, cv2.MORPH_OPEN, k_v)
    scratch_mask |= (opened_v > 80)

    return scratch_mask.astype(np.float32)


def auto_detect_damage_mask(image: np.ndarray) -> np.ndarray:
    """Return binary damage mask: 1.0 where damaged, 0.0 where valid."""
    valid_mask, _ = detect_damage_mask(image)
    return (valid_mask < 0.5).astype(np.float32)
