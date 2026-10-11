"""
Background restoration service worker orchestrating DIP pipeline execution.
"""
import time
import traceback
from typing import Optional
import numpy as np
import torch

from dip_core.config import DipTrainingConfig, RestorationTask, DamageConfig
from dip_core.pipeline.restoration_pipeline import run_restoration
from dip_core.pipeline.mask_detection import auto_detect_damage_mask, detect_damage_mask
from dip_core.pipeline.damage_simulation import simulate_thermal_damage
from backend.app.services.job_store import job_store
from backend.app.services.image_io import array_to_base64_png
from backend.app.config import DEVICE

def execute_restoration_job(
    job_id: str,
    image_arr: np.ndarray,
    mask_arr: Optional[np.ndarray],
    options: dict
) -> None:
    """
    Worker function invoked asynchronously in a background thread.
    """
    job = job_store.get_job(job_id)
    if not job:
        return
        
    cancel_event = job_store.get_cancel_event(job_id)
    
    # Ensure sequential GPU execution
    with job_store.execution_lock:
        if cancel_event and cancel_event.is_set():
            return
            
        try:
            task_str = options.get("task", "auto")
            model_str = options.get("model", "unet")
            total_iters = int(options.get("iterations", 1500))
            lr = float(options.get("learning_rate", 0.01))
            reg_noise_std = float(options.get("reg_noise_std", 0.03))
            colormap_str = options.get("colormap", "ironbow")
            simulate_damage_flag = bool(options.get("simulate_damage", False))
            auto_detect_mask_flag = bool(options.get("auto_detect_mask", True))
            early_stopping_flag = bool(options.get("early_stopping", True))
            patience = int(options.get("patience", 250))
            
            ground_truth_arr = image_arr.copy()
            degraded_arr = image_arr.copy()
            
            # Step 1: Simulate degradation if requested
            if simulate_damage_flag:
                dmg_cfg = DamageConfig(
                    dead_pixel_ratio=0.015,
                    scratch_line_count=2,
                    scratch_line_thickness=2,
                    seed=42
                )
                degraded_arr, sim_mask = simulate_thermal_damage(ground_truth_arr, dmg_cfg)
                if mask_arr is None:
                    mask_arr = sim_mask
                if task_str == "auto":
                    task_str = "inpaint"
            elif (auto_detect_mask_flag or task_str == "auto") and mask_arr is None:
                # Automatic defect and dead-pixel detection
                valid_mask, stats = detect_damage_mask(degraded_arr)
                if stats.get("damaged_pixels", 0) > 0:
                    mask_arr = valid_mask
                    if task_str == "auto":
                        task_str = "inpaint"
                else:
                    mask_arr = np.ones_like(degraded_arr, dtype=np.float32)
                    if task_str == "auto":
                        task_str = "denoise"
                
            if mask_arr is None:
                mask_arr = np.ones_like(degraded_arr, dtype=np.float32)

            # Store degraded, mask, original base64
            orig_b64 = array_to_base64_png(ground_truth_arr, colormap=colormap_str)
            deg_b64 = array_to_base64_png(degraded_arr, colormap=colormap_str)
            mask_b64 = array_to_base64_png(mask_arr, colormap="grayscale")
            
            with job_store._lock:
                job["original_image_base64"] = orig_b64
                job["degraded_image_base64"] = deg_b64
                job["mask_image_base64"] = mask_b64
                
            # Progress callback
            last_preview_time = [time.time()]
            
            def on_step_callback(step: int, loss: float, preview_tensor):
                now = time.time()
                # Generate preview thumbnail every 100 steps or at least 2 seconds apart
                preview_b64 = None
                if (now - last_preview_time[0] >= 2.0 or step == total_iters) and preview_tensor is not None:
                    last_preview_time[0] = now
                    if hasattr(preview_tensor, "detach"):
                        prev_np = preview_tensor.detach().squeeze().cpu().numpy()
                    else:
                        prev_np = np.asarray(preview_tensor).squeeze()
                    preview_b64 = array_to_base64_png(prev_np, colormap=colormap_str)
                    
                elapsed = now - start_time
                avg_iter_time = elapsed / max(step, 1)
                eta = (total_iters - step) * avg_iter_time
                
                job_store.update_progress(
                    job_id=job_id,
                    current_iter=step,
                    total_iter=total_iters,
                    loss=loss,
                    preview_base64=preview_b64,
                    eta_seconds=eta
                )
                
            train_cfg = DipTrainingConfig(
                iterations=total_iters,
                learning_rate=lr,
                reg_noise_std=reg_noise_std,
                save_every=50,
                log_every=50,
                early_stopping_patience=patience,
                early_stopping_enabled=early_stopping_flag,
                device=DEVICE
            )
            
            start_time = time.time()
            
            # Map task string to enum
            if task_str in ("inpainting", "inpaint"):
                task_enum = RestorationTask.INPAINT
            elif task_str in ("denoising", "denoise"):
                task_enum = RestorationTask.DENOISE
            elif task_str in ("super_resolution", "sr"):
                task_enum = RestorationTask.SUPER_RESOLUTION
            elif task_str in ("combined", "denoise_and_inpaint"):
                task_enum = RestorationTask.DENOISE_AND_INPAINT
            else:
                task_enum = RestorationTask.AUTO
            
            output = run_restoration(
                image_input=degraded_arr,
                mask_input=mask_arr,
                task=task_enum,
                model_name=model_str,
                training_config=train_cfg,
                cancel_event=cancel_event,
                step_callback=on_step_callback,
                ground_truth=ground_truth_arr if simulate_damage_flag else None
            )
            
            if cancel_event and cancel_event.is_set():
                job_store.cancel_job(job_id)
                return
                
            restored_b64 = array_to_base64_png(output.restored_array, colormap=colormap_str)
            
            job_store.complete_job(
                job_id=job_id,
                restored_base64=restored_b64,
                original_base64=orig_b64,
                degraded_base64=deg_b64,
                mask_base64=mask_b64,
                metrics=output.metrics,
                losses=output.losses,
                runtime_seconds=output.runtime_seconds
            )
            
        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            job_store.fail_job(job_id, err_msg)
