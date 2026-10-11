"""Synthetic thermal image damage simulation for objective benchmarking."""

from typing import Tuple, Optional
import numpy as np
import cv2
from dip_core.config import DamageConfig


def simulate_damage(
    clean_image: np.ndarray,
    config: Optional[DamageConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate realistic synthetic damage on a clean thermal image.

    Features:
      1. Gaussian thermal sensor noise with configurable SNR in dB.
      2. Pixel dropout (dead pixels).
      3. Structural defects: rectangular missing sensor blocks and diagonal scratch lines.

    Args:
      clean_image: float32 numpy array [0.0, 1.0] of shape (H, W).
      config: DamageConfig containing SNR, dropout ratio, line holes count, etc.

    Returns:
      damaged_image: degraded float32 numpy array in [0.0, 1.0].
      ground_truth_mask: binary mask M (1.0 = valid, 0.0 = damaged/missing).
      clean_reference: pristine clean float32 reference image.
    """
    if config is None:
        config = DamageConfig()

    rng = np.random.default_rng(config.seed)
    clean = np.clip(clean_image.copy(), 0.0, 1.0).astype(np.float32)
    h, w = clean.shape[:2]

    damaged = clean.copy()
    mask = np.ones((h, w), dtype=np.float32)

    # 1. Additive Gaussian Sensor Noise
    if config.gaussian_snr_db is not None:
        signal_power = np.mean(clean**2)
        noise_power = signal_power / (10.0 ** (config.gaussian_snr_db / 10.0))
        noise_std = float(np.sqrt(max(noise_power, 1e-12)))
    else:
        noise_std = config.gaussian_noise_std

    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, (h, w)).astype(np.float32)
        damaged = np.clip(damaged + noise, 0.0, 1.0)

    # 2. Random Pixel Dropout (Dead pixels)
    if config.dropout_ratio > 0:
        drop_mask = rng.uniform(0.0, 1.0, (h, w)) < config.dropout_ratio
        damaged[drop_mask] = 0.0
        mask[drop_mask] = 0.0

    # 3. Rectangular Missing Holes (Sensor Block Failure)
    for _ in range(config.num_rect_holes):
        bh = int(h * rng.uniform(0.12, 0.22))
        bw = int(w * rng.uniform(0.12, 0.22))
        y0 = int(rng.integers(0, max(1, h - bh)))
        x0 = int(rng.integers(0, max(1, w - bw)))
        damaged[y0 : y0 + bh, x0 : x0 + bw] = 0.0
        mask[y0 : y0 + bh, x0 : x0 + bw] = 0.0

    # 4. Scratch Lines
    for _ in range(config.num_line_holes):
        y1, x1 = int(rng.integers(0, h)), int(rng.integers(0, w))
        y2, x2 = int(rng.integers(0, h)), int(rng.integers(0, w))
        thickness = int(rng.integers(2, 4))
        cv2.line(mask, (x1, y1), (x2, y2), 0.0, thickness=thickness)

    # Apply structural mask to damaged image (damaged areas have 0 pixel value)
    damaged = damaged * mask

    return damaged.astype(np.float32), mask.astype(np.float32), clean


def simulate_thermal_damage(
    image: np.ndarray,
    config: Optional[DamageConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper returning (damaged_image, mask)."""
    damaged, mask, _ = simulate_damage(image, config)
    return damaged, mask


def add_gaussian_noise(image: np.ndarray, snr_db: float = 20.0, seed: int = 42) -> np.ndarray:
    cfg = DamageConfig(gaussian_snr_db=snr_db, dropout_ratio=0.0, num_rect_holes=0, num_line_holes=0, seed=seed)
    damaged, _, _ = simulate_damage(image, cfg)
    return damaged


def add_dead_pixels(image: np.ndarray, ratio: float = 0.02, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    cfg = DamageConfig(gaussian_snr_db=None, gaussian_noise_std=0.0, dropout_ratio=ratio, num_rect_holes=0, num_line_holes=0, seed=seed)
    damaged, mask, _ = simulate_damage(image, cfg)
    return damaged, mask


def add_rectangular_holes(image: np.ndarray, count: int = 2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    cfg = DamageConfig(gaussian_snr_db=None, gaussian_noise_std=0.0, dropout_ratio=0.0, num_rect_holes=count, num_line_holes=0, seed=seed)
    damaged, mask, _ = simulate_damage(image, cfg)
    return damaged, mask


def add_scratch_lines(image: np.ndarray, count: int = 2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    cfg = DamageConfig(gaussian_snr_db=None, gaussian_noise_std=0.0, dropout_ratio=0.0, num_rect_holes=0, num_line_holes=count, seed=seed)
    damaged, mask, _ = simulate_damage(image, cfg)
    return damaged, mask
