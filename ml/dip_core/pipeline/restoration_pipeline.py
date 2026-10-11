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
from dip_core.pipeline.image_postprocessing import composite_restored_image, enhance_sharpness_and_contrast
from dip_core.pipeline.deblurring import (
    compute_sharpness_metrics,
    strong_deblur_pipeline,
    multi_scale_detail_recovery,
)
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

    @property
    def restored_array(self) -> np.ndarray:
        return self.restored_composited

    @property
    def losses(self) -> List[float]:
        return self.loss_history

    @property
    def runtime_seconds(self) -> float:
        return self.duration_seconds


def run_restoration(
    image: Optional[Union[str, Path, bytes, np.ndarray]] = None,
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
    # Backward compatible kwargs
    image_input: Optional[Union[str, Path, bytes, np.ndarray]] = None,
    mask_input: Optional[Union[str, Path, bytes, np.ndarray]] = None,
    training_config: Optional[DipTrainingConfig] = None,
    ground_truth: Optional[Union[str, Path, bytes, np.ndarray]] = None,
    step_callback: Optional[Callable] = None,
    **kwargs
) -> RestorationOutput:
    """
    Unified entry point executing full thermal restoration and comparative evaluation.
    """
    if image is None and image_input is not None:
        image = image_input
    if mask is None and mask_input is not None:
        mask = mask_input
    if reference_image is None and ground_truth is not None:
        reference_image = ground_truth

    if training_config is not None:
        iterations = training_config.iterations
        learning_rate = training_config.learning_rate
        reg_noise_std = training_config.reg_noise_std
        if training_config.device is not None:
            device = training_config.device

    if step_callback is not None and progress_callback is None:
        def bridged_cb(it, total, loss, prev):
            step_callback(it, loss, prev)
        progress_callback = bridged_cb

    if hasattr(task, "value"):
        task = task.value
    if task == "inpaint":
        task = "inpaint"

    dev = get_device(device)

    # 1. Load input image
    raw_img = load_image(image, grayscale=True)
    orig_shape = raw_img.shape[:2]

    # Clean reference tracking
    clean_ref = None
    if reference_image is not None:
        clean_ref = load_image(reference_image, grayscale=True)

    # 2. Damage simulation (if requested for benchmarking clean thermal images)
    # 2. Damage simulation or degradation analysis
    if simulate_damage_flag:
        clean_ref = raw_img.copy()
        degraded_img, sim_mask, _ = simulate_damage(clean_ref, damage_config)
        mask_np = sim_mask
    else:
        degraded_img = raw_img.copy()

    # Initial sharpness measurement
    initial_sharpness = compute_sharpness_metrics(degraded_img)

    if not simulate_damage_flag:
        if task == "auto" or auto_detect_mask:
            det_mask, stats = detect_damage_mask(degraded_img)
            if mask is not None:
                mask_tensor_tmp = build_valid_mask(mask, target_shape=orig_shape, task="inpaint", device="cpu")
                mask_np = mask_tensor_tmp.squeeze().numpy()
                task = "inpaint"
            elif stats.get("damaged_pixels", 0) > 20:
                mask_np = det_mask
                task = "inpaint"
            elif initial_sharpness["is_blurry"]:
                mask_np = np.ones(orig_shape, dtype=np.float32)
                task = "deblur"
            else:
                mask_np = np.ones(orig_shape, dtype=np.float32)
                task = "denoise"
        elif task in ("deblur", "deblurring", "sharpen"):
            mask_np = np.ones(orig_shape, dtype=np.float32)
            task = "deblur"
        elif mask is not None:
            # Build valid mask from user input
            mask_tensor_tmp = build_valid_mask(mask, target_shape=orig_shape, task=task, device="cpu")
            mask_np = mask_tensor_tmp.squeeze().numpy()
        else:
            mask_np = np.ones(orig_shape, dtype=np.float32)

    # 3. Physics-informed Deblurring Prior Generation
    deblur_prior = None
    if task in ("deblur", "deblurring", "sharpen") or (task in ("auto", "denoise") and initial_sharpness["is_blurry"]):
        strength_setting = getattr(training_config, "deblur_strength", "strong") if training_config else "strong"
        deblur_prior, _ = strong_deblur_pipeline(degraded_img, strength=strength_setting)

    # 4. Convert to torch tensors and pad to multiple of 32
    target_train_img = deblur_prior if (deblur_prior is not None and task == "deblur") else degraded_img
    deg_t = to_tensor(target_train_img, device=dev)
    deg_padded, orig_size = pad_to_multiple_of_32(deg_t)

    mask_t = to_tensor(mask_np, device=dev)
    mask_padded, _ = pad_to_multiple_of_32(mask_t)

    ref_padded = None
    if clean_ref is not None:
        ref_t = to_tensor(clean_ref, device=dev)
        ref_padded, _ = pad_to_multiple_of_32(ref_t)

    # 5. Instantiate Generator Architecture
    if model_name == "conv_baseline":
        model = ConvBaselineGenerator(in_channels=32, out_channels=1)
    else:
        model = UNetGenerator(in_channels=32, out_channels=1, skip_channels=4)

    # 6. Training loop configuration
    cfg = DipTrainingConfig(
        iterations=iterations,
        learning_rate=learning_rate,
        reg_noise_std=reg_noise_std,
        device=str(dev),
        loss_type="composite_edge",
    )
    if training_config is not None:
        cfg.loss_type = getattr(training_config, "loss_type", "composite_edge")
        cfg.deblur_strength = getattr(training_config, "deblur_strength", "strong")

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

    # 7. Execute DIP optimization
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

    # 8. Adaptive Convergence Detail Fusion & Post-Processing
    final_loss_val = train_result.loss_history[-1] if train_result.loss_history else 0.05
    if deblur_prior is not None:
        # If UNet reached deep convergence (<0.005 loss), blend its regularized structure.
        # If under-converged (e.g. CPU fast run), lean primarily on the sharp deblur prior.
        dip_weight = float(np.clip((0.005 - final_loss_val) / 0.005, 0.0, 0.4))
        fused = (1.0 - dip_weight) * deblur_prior + dip_weight * restored_raw
        restored_raw = np.clip(fused, 0.0, 1.0).astype(np.float32)
    elif task == "denoise" or np.all(mask_np >= 0.999):
        # Eliminate DIP over-smoothing for denoising
        restored_raw = enhance_sharpness_and_contrast(restored_raw, strength=1.1, clahe_clip=1.3)

    # Blend with original valid pixels if inpainting
    if task in ("denoise", "deblur", "sharpen") or np.all(mask_np >= 0.999):
        restored_composited = restored_raw
    else:
        restored_composited = composite_restored_image(degraded_img, restored_raw, mask_np)

    # 9. Evaluate Classical Baselines
    baseline_eval = evaluate_baselines(
        degraded=degraded_img,
        mask=mask_np,
        reference=clean_ref,
        task=task,
    )

    # 10. Compute final metrics
    final_sharpness = compute_sharpness_metrics(restored_composited)
    orig_lap = max(initial_sharpness["laplacian_variance"], 1e-4)
    orig_ten = max(initial_sharpness["tenengrad_energy"], 1e-4)

    metrics: Dict[str, Any] = {
        "final_loss": train_result.loss_history[-1] if train_result.loss_history else 0.0,
        "best_iteration": train_result.best_iteration,
        "total_iterations": len(train_result.loss_history),
        "duration_seconds": train_result.duration_seconds,
        "stopped_early": train_result.stopped_early,
        "sharpness_before": initial_sharpness["laplacian_variance"],
        "sharpness_after": final_sharpness["laplacian_variance"],
        "sharpness_gain_pct": round(((final_sharpness["laplacian_variance"] / orig_lap) - 1.0) * 100.0, 1),
        "tenengrad_before": initial_sharpness["tenengrad_energy"],
        "tenengrad_after": final_sharpness["tenengrad_energy"],
        "tenengrad_gain_pct": round(((final_sharpness["tenengrad_energy"] / orig_ten) - 1.0) * 100.0, 1),
        "is_blurry_before": initial_sharpness["is_blurry"],
        "is_blurry_after": final_sharpness["is_blurry"],
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
