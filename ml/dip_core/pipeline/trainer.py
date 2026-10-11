import time
from typing import Callable, Dict, Any, Optional, Union
import numpy as np
import torch
import torch.optim as optim

from dip_core.config import DIPConfig
from dip_core.models.unet import UNetDIP
from dip_core.models.conv_baseline import ConvDIPBaseline
from dip_core.pipeline.preprocess import (
    load_image,
    pad_to_multiple,
    to_torch_tensor,
    build_mask,
    get_device,
)
from dip_core.pipeline.noise_input import get_noise_input, perturb_noise
from dip_core.pipeline.losses import MaskedMSELoss
from dip_core.pipeline.early_stopping import EarlyStoppingPlateau
from dip_core.pipeline.postprocess import tensor_to_numpy
from dip_core.metrics import calculate_psnr, calculate_ssim
from dip_core.baselines import evaluate_baselines


def train_dip(
    image_input: Union[str, np.ndarray, Any],
    mask_input: Optional[Union[str, np.ndarray, Any]] = None,
    reference_input: Optional[Union[str, np.ndarray, Any]] = None,
    config: Optional[DIPConfig] = None,
    task: str = "denoise",
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Execute Deep Image Prior optimization loop on a single thermal image.

    Features:
      - Fixed noise input z ~ N(0, I) with 32 channels.
      - Regularization noise jitter (reg_noise_std = 1/30) added to z each iteration.
      - Adam optimizer (lr = 0.01).
      - Periodic evaluation every 50 iterations (PSNR and SSIM when reference is provided).
      - Tracks and preserves best output (lowest smoothed loss or highest PSNR).
      - Early stopping on moving-average loss plateau (optional flag).
      - Real-time progress callback.
    """
    if config is None:
        config = DIPConfig()

    device = get_device(config.device)

    # 1. Load image and reference
    degraded_np = load_image(image_input, grayscale=True)
    orig_shape = degraded_np.shape[:2]
    ref_np = load_image(reference_input, grayscale=True) if reference_input is not None else None

    # 2. Convert to tensors and pad to multiple of 32
    target_tensor = to_torch_tensor(degraded_np, device=device)
    target_padded, orig_size = pad_to_multiple(target_tensor, multiple=32)
    padded_h, padded_w = target_padded.shape[-2], target_padded.shape[-1]

    # Build mask (1 = valid, 0 = missing)
    mask_tensor = build_mask(
        mask_input=mask_input,
        target_shape=orig_shape,
        task=task,
        device=device,
    )
    mask_padded, _ = pad_to_multiple(mask_tensor, multiple=32)

    # 3. Model selection
    if config.model_type == "conv_baseline":
        model = ConvDIPBaseline(
            in_channels=config.input_channels,
            out_channels=config.out_channels,
        ).to(device)
    else:
        model = UNetDIP(
            in_channels=config.input_channels,
            out_channels=config.out_channels,
            skip_channels=config.skip_channels,
        ).to(device)

    # 4. Generate fixed input noise z ~ N(0, I) with 32 channels
    z_fixed = get_noise_input(
        spatial_shape=(padded_h, padded_w),
        channels=config.input_channels,
        device=device,
        std=config.noise_std,
    )

    # 5. Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=False,
    )
    criterion = MaskedMSELoss()

    early_stopper = EarlyStoppingPlateau(
        patience=config.patience,
        min_delta=config.min_delta,
        min_iterations=min(300, config.iterations // 2),
    )

    loss_history = []
    metric_history = []
    best_output_tensor = None
    best_score = float("inf") if ref_np is None else -float("inf")
    best_iteration = 0
    start_time = time.time()
    stopped_early = False

    # 6. Optimization Loop
    for it in range(1, config.iterations + 1):
        if is_cancelled and is_cancelled():
            return {
                "status": "cancelled",
                "iteration": it,
                "loss_history": loss_history,
                "metric_history": metric_history,
                "message": "Optimization cancelled by user.",
            }

        optimizer.zero_grad()

        # Regularization: small additive noise to z (reg_noise_std = 1/30)
        z_in = perturb_noise(z_fixed, reg_noise_std=config.reg_noise_std)

        # Forward pass
        pred = model(z_in)

        # Masked loss calculation
        loss = criterion(pred, target_padded, mask_padded)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        loss_history.append(round(loss_val, 6))

        # Check early stopping
        early_stop_triggered = False
        if config.early_stopping:
            early_stop_triggered = early_stopper.step(
                loss=loss_val,
                iteration=it,
                model=model,
                current_output=pred,
            )

        # Checkpoint best output
        with torch.no_grad():
            if ref_np is not None:
                # If reference exists: track best PSNR
                cur_img = tensor_to_numpy(pred, orig_size)
                cur_psnr = calculate_psnr(cur_img, ref_np)
                if cur_psnr > best_score:
                    best_score = cur_psnr
                    best_iteration = it
                    best_output_tensor = pred.detach().clone()
            else:
                # If no reference: track lowest loss
                if loss_val < best_score:
                    best_score = loss_val
                    best_iteration = it
                    best_output_tensor = pred.detach().clone()

        # Periodic logging every 50 iterations or at the end
        if it % config.log_interval == 0 or it == 1 or it == config.iterations or early_stop_triggered:
            elapsed = time.time() - start_time
            with torch.no_grad():
                cur_img = tensor_to_numpy(pred, orig_size)

            log_entry = {
                "iteration": it,
                "loss": round(loss_val, 6),
                "elapsed_sec": round(elapsed, 1),
            }

            if ref_np is not None:
                psnr_val = round(calculate_psnr(cur_img, ref_np), 2)
                ssim_val = round(calculate_ssim(cur_img, ref_np), 4)
                log_entry["psnr"] = psnr_val
                log_entry["ssim"] = ssim_val
            else:
                # Proxy PSNR against valid degraded pixels
                m_np = tensor_to_numpy(mask_tensor, orig_size) if task == "inpaint" else np.ones_like(degraded_np)
                psnr_val = round(calculate_psnr(cur_img * m_np, degraded_np * m_np), 2)
                log_entry["psnr_proxy"] = psnr_val

            metric_history.append(log_entry)

            if on_progress:
                on_progress({
                    "iteration": it,
                    "total_iterations": config.iterations,
                    "progress_pct": round((it / config.iterations) * 100, 1),
                    "loss": round(loss_val, 6),
                    "best_score": round(best_score, 4),
                    "psnr": log_entry.get("psnr", log_entry.get("psnr_proxy", 0.0)),
                    "ssim": log_entry.get("ssim"),
                    "elapsed_sec": round(elapsed, 1),
                    "preview_image": cur_img,
                })

        if early_stop_triggered:
            stopped_early = True
            break

    # 7. Final restored output from best checkpoint
    if best_output_tensor is not None:
        final_tensor = best_output_tensor
    else:
        with torch.no_grad():
            final_tensor = model(z_fixed)

    final_image = tensor_to_numpy(final_tensor, orig_size)
    total_time = time.time() - start_time

    # Final metrics
    final_metrics = {
        "final_loss": round(loss_history[-1] if loss_history else 0.0, 6),
        "best_iteration": best_iteration,
        "total_iterations": it,
        "stopped_early": stopped_early,
        "elapsed_seconds": round(total_time, 2),
    }

    if ref_np is not None:
        final_metrics["psnr"] = round(calculate_psnr(final_image, ref_np), 2)
        final_metrics["ssim"] = round(calculate_ssim(final_image, ref_np), 4)
    else:
        m_np = tensor_to_numpy(mask_tensor, orig_size) if task == "inpaint" else np.ones_like(degraded_np)
        final_metrics["psnr_proxy"] = round(calculate_psnr(final_image * m_np, degraded_np * m_np), 2)

    # 8. Baselines
    baselines_results = {}
    if config.run_baselines:
        mask_2d = tensor_to_numpy(mask_tensor, orig_size) if task == "inpaint" else None
        baselines_results = evaluate_baselines(
            degraded=degraded_np,
            mask=mask_2d,
            reference=ref_np,
            task=task,
        )

    return {
        "status": "completed",
        "task": task,
        "model_type": config.model_type,
        "restored_image": final_image,
        "degraded_image": degraded_np,
        "reference_image": ref_np,
        "metrics": final_metrics,
        "loss_history": loss_history,
        "metric_history": metric_history,
        "baselines": baselines_results,
    }
