"""Unified restoration pipeline: single entry point for backend, CLI, and notebooks."""

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Dict, Any, Union, List
import numpy as np
import torch

from dip_core.config import DipTrainingConfig, DamageConfig, get_device
from dip_core.models.unet_generator import UNetGenerator
from dip_core.models.conv_baseline_generator import ConvBaselineGenerator
from dip_core.pipeline.image_preprocessing import (
    load_image,
    pad_to_multiple_of_32,
    crop_back_to_original_size,
    to_tensor,
    build_valid_mask,
)
from dip_core.pipeline.damage_simulation import simulate_damage
from dip_core.pipeline.mask_detection import detect_damage_mask
from dip_core.pipeline.dip_trainer import DipTrainer
from dip_core.pipeline.image_postprocessing import composite_restored_image
from dip_core.metrics import calculate_psnr, calculate_ssim, calculate_masked_psnr
from dip_core.baselines import evaluate_baselines


@dataclass
class RestorationOutput:
    """Comprehensive output payload returned by run_restoration."""
    original_image: np.ndarray
    degraded_image: np.ndarray
    mask: np.ndarray
    restored_raw: np.ndarray
    restored_composited: np.ndarray
    baseline_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    loss_history: List[float] = field(default_factory=list)
    psnr_history: List[float] = field(default_factory=list)
    ssim_history: List[float] = field(default_factory=list)
    best_iteration: int = 0
    duration_seconds: float = 0.0
    task: str = "denoise"
    model_name: str = "unet"
    stopped_early: bool = False
    cancelled: bool = False


def run_restoration(
    image: Union[str, Path, bytes, np.ndarray],
    mask: Optional[Union[str, Path, bytes, np.ndarray]] = None,
    task: str = "denoise",
    model_name: str = "unet",
    iterations: int = 1500,
    learning_rate: float = 0.01,
    reg_noise_std: float = 1.0 / 30.0,
    auto_detect_mask: bool = False,
    simulate_damage_flag: bool = False,
    damage_config: Optional[DamageConfig] = None,
    reference_image: Optional[Union[str, Path, bytes, np.ndarray]] = None,
    progress_callback: Optional[Callable[[int, int, float, Optional[np.ndarray]], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    device: Optional[str] = None,
) -> RestorationOutput:
    """
    Unified entry point executing full thermal restoration and comparative evaluation.
    """
    dev = get_device(device)

    # 1. Load input image
    raw_img = load_image(image, grayscale=True)
    orig_shape = raw_img.shape[:2]

    # Clean reference tracking
    clean_ref = None
    if reference_image is not None:
        clean_ref = load_image(reference_image, grayscale=True)

    # 2. Damage simulation (if requested for benchmarking clean thermal images)
    if simulate_damage_flag:
        clean_ref = raw_img.copy()
        degraded_img, sim_mask, _ = simulate_damage(clean_ref, damage_config)
        mask_np = sim_mask
    else:
        degraded_img = raw_img.copy()
        if auto_detect_mask:
            mask_np, _ = detect_damage_mask(degraded_img)
        elif mask is not None:
            # Build valid mask from user input
            mask_tensor_tmp = build_valid_mask(mask, target_shape=orig_shape, task=task, device="cpu")
            mask_np = mask_tensor_tmp.squeeze().numpy()
        else:
            mask_np = np.ones(orig_shape, dtype=np.float32)

    # 3. Convert to torch tensors and pad to multiple of 32
    deg_t = to_tensor(degraded_img, device=dev)
    deg_padded, orig_size = pad_to_multiple_of_32(deg_t)

    mask_t = to_tensor(mask_np, device=dev)
    mask_padded, _ = pad_to_multiple_of_32(mask_t)

    ref_padded = None
    if clean_ref is not None:
        ref_t = to_tensor(clean_ref, device=dev)
        ref_padded, _ = pad_to_multiple_of_32(ref_t)

    # 4. Instantiate Generator Architecture
    if model_name == "conv_baseline":
        model = ConvBaselineGenerator(in_channels=32, out_channels=1)
    else:
        model = UNetGenerator(in_channels=32, out_channels=1, skip_channels=4)

    # 5. Training loop configuration
    cfg = DipTrainingConfig(
        iterations=iterations,
        learning_rate=learning_rate,
        reg_noise_std=reg_noise_std,
        device=str(dev),
    )

    trainer = DipTrainer(config=cfg)

    # Wrapped progress callback unpadding preview
    def wrapped_cb(it, total, loss, preview_np):
        if progress_callback and preview_np is not None:
            # Unpad preview array back to original dimensions
            preview_t = torch.from_numpy(preview_np).unsqueeze(0).unsqueeze(0)
            unpadded_preview = crop_back_to_original_size(preview_t, orig_size).squeeze().numpy()
            progress_callback(it, total, loss, unpadded_preview)
        elif progress_callback:
            progress_callback(it, total, loss, None)

    # 6. Execute DIP optimization
    train_result = trainer.train(
        degraded_tensor=deg_padded,
        mask_tensor=mask_padded,
        model=model,
        config=cfg,
        progress_callback=wrapped_cb,
        reference_tensor=ref_padded,
        cancel_event=cancel_event,
    )

    # Unpad restored raw output
    res_t = torch.from_numpy(train_result.restored_image).unsqueeze(0).unsqueeze(0)
    restored_raw = crop_back_to_original_size(res_t, orig_size).squeeze().numpy()

    # Create composited blend: mask * degraded + (1 - mask) * restored
    restored_composited = composite_restored_image(degraded_img, restored_raw, mask_np)

    # 7. Evaluate Classical Baselines
    baseline_eval = evaluate_baselines(
        degraded=degraded_img,
        mask=mask_np,
        reference=clean_ref,
        task=task,
    )

    # 8. Compute final metrics
    metrics: Dict[str, Any] = {
        "final_loss": train_result.loss_history[-1] if train_result.loss_history else 0.0,
        "best_iteration": train_result.best_iteration,
        "total_iterations": len(train_result.loss_history),
        "duration_seconds": train_result.duration_seconds,
        "stopped_early": train_result.stopped_early,
    }

    if clean_ref is not None:
        metrics["psnr"] = round(calculate_psnr(restored_composited, clean_ref), 2)
        metrics["psnr_raw"] = round(calculate_psnr(restored_raw, clean_ref), 2)
        metrics["ssim"] = round(calculate_ssim(restored_composited, clean_ref), 4)
        metrics["ssim_raw"] = round(calculate_ssim(restored_raw, clean_ref), 4)
        metrics["masked_psnr"] = round(calculate_masked_psnr(restored_composited, clean_ref, mask_np), 2)
        metrics["degraded_psnr"] = round(calculate_psnr(degraded_img, clean_ref), 2)
        metrics["degraded_ssim"] = round(calculate_ssim(degraded_img, clean_ref), 4)
    else:
        # Proxy PSNR against valid degraded pixels
        metrics["psnr_proxy"] = round(calculate_psnr(restored_composited * mask_np, degraded_img * mask_np), 2)

    return RestorationOutput(
        original_image=raw_img,
        degraded_image=degraded_img,
        mask=mask_np,
        restored_raw=restored_raw,
        restored_composited=restored_composited,
        baseline_results=baseline_eval,
        metrics=metrics,
        loss_history=train_result.loss_history,
        psnr_history=train_result.psnr_history,
        ssim_history=train_result.ssim_history,
        best_iteration=train_result.best_iteration,
        duration_seconds=train_result.duration_seconds,
        task=task,
        model_name=model_name,
        stopped_early=train_result.stopped_early,
        cancelled=train_result.cancelled,
    )
