"""Deep Image Prior optimization loop and trainer engine."""

import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any
import numpy as np
import torch
import torch.optim as optim

from dip_core.config import DipTrainingConfig, get_device
from dip_core.pipeline.noise_input import create_fixed_noise, perturb_noise
from dip_core.pipeline.masked_loss import MaskedMSELoss, CompositeEdgeLoss
from dip_core.pipeline.early_stopping import EarlyStopper
from dip_core.metrics import calculate_psnr, calculate_ssim


@dataclass
class TrainingResult:
    """Dataclass holding output artifacts from DIP optimization."""
    restored_image: np.ndarray
    best_iteration: int
    loss_history: List[float] = field(default_factory=list)
    psnr_history: List[float] = field(default_factory=list)
    ssim_history: List[float] = field(default_factory=list)
    duration_seconds: float = 0.0
    stopped_early: bool = False
    cancelled: bool = False


class DipTrainer:
    """
    Orchestrates Deep Image Prior network optimization on a single thermal image.
    """

    def __init__(self, config: Optional[DipTrainingConfig] = None):
        self.config = config or DipTrainingConfig()

    def train(
        self,
        degraded_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
        model: torch.nn.Module,
        config: Optional[DipTrainingConfig] = None,
        progress_callback: Optional[Callable[[int, int, float, Optional[np.ndarray]], None]] = None,
        reference_tensor: Optional[torch.Tensor] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> TrainingResult:
        """
        Execute the optimization loop.

        Args:
          degraded_tensor: (1, 1, H, W) float tensor padded to multiple of 32.
          mask_tensor: (1, 1, H, W) float binary mask (1=valid, 0=damaged).
          model: generator module (UNetGenerator or ConvBaselineGenerator).
          config: training hyperparameters.
          progress_callback: fn(iteration, total_iters, loss, preview_np).
          reference_tensor: optional (1, 1, H, W) clean target for objective tracking.
          cancel_event: threading.Event checked every step for cancellation.

        Returns:
          TrainingResult containing the restored image array and performance curves.
        """
        cfg = config or self.config
        device = get_device(cfg.device)

        model = model.to(device)
        degraded_tensor = degraded_tensor.to(device)
        mask_tensor = mask_tensor.to(device)
        if reference_tensor is not None:
            reference_tensor = reference_tensor.to(device)

        h, w = degraded_tensor.shape[-2], degraded_tensor.shape[-1]

        # 1. Fixed input noise z ~ N(0, I)
        z_fixed = create_fixed_noise(
            spatial_shape=(h, w),
            channels=cfg.noise_channels,
            seed=cfg.seed,
            device=device,
        )

        # 2. Adam optimizer (foreach=False for host resilience)
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            foreach=False,
        )
        if getattr(cfg, "loss_type", "composite_edge") == "composite_edge":
            criterion = CompositeEdgeLoss().to(device)
        else:
            criterion = MaskedMSELoss().to(device)
        early_stopper = EarlyStopper(
            patience=cfg.early_stopping_patience,
            min_delta=cfg.early_stopping_min_delta,
            smoothing_window=cfg.smoothing_window,
        )

        loss_history: List[float] = []
        psnr_history: List[float] = []
        ssim_history: List[float] = []

        best_output = None
        best_score = float("inf") if reference_tensor is None else -float("inf")
        best_iteration = 0
        stopped_early = False
        cancelled = False
        start_time = time.time()

        model.train()

        try:
            for it in range(1, cfg.iterations + 1):
                # Cancellation check
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break

                optimizer.zero_grad()

                # Regularization jitter
                z_in = perturb_noise(z_fixed, reg_noise_std=cfg.reg_noise_std)

                # Forward pass
                pred = model(z_in)

                # Masked reconstruction loss
                loss = criterion(pred, degraded_tensor, mask_tensor)
                loss.backward()
                optimizer.step()

                loss_val = float(loss.item())
                loss_history.append(round(loss_val, 6))

                # Early stopping check
                if cfg.early_stopping_enabled:
                    if early_stopper.step(loss_val, it, current_output=pred):
                        stopped_early = True
                        break

                # Best output tracking
                with torch.no_grad():
                    if reference_tensor is not None:
                        # PSNR-driven tracking when ground truth exists
                        cur_psnr = calculate_psnr(pred, reference_tensor)
                        if cur_psnr > best_score:
                            best_score = cur_psnr
                            best_iteration = it
                            best_output = pred.detach().cpu().clone()
                    else:
                        # Loss-driven tracking when unsupervised
                        if loss_val < best_score:
                            best_score = loss_val
                            best_iteration = it
                            best_output = pred.detach().cpu().clone()

                # Periodic progress logging
                if it % cfg.log_every == 0 or it == 1 or it == cfg.iterations:
                    preview_np = None
                    with torch.no_grad():
                        pred_np = pred.detach().cpu().squeeze().numpy()
                        preview_np = np.clip(pred_np, 0.0, 1.0)

                        if reference_tensor is not None:
                            ref_np = reference_tensor.detach().cpu().squeeze().numpy()
                            p_val = round(calculate_psnr(pred_np, ref_np), 2)
                            s_val = round(calculate_ssim(pred_np, ref_np), 4)
                            psnr_history.append(p_val)
                            ssim_history.append(s_val)

                    if progress_callback:
                        progress_callback(it, cfg.iterations, round(loss_val, 6), preview_np)

            # Retrieve final restored image
            if best_output is not None:
                final_tensor = best_output
            else:
                with torch.no_grad():
                    model.eval()
                    final_tensor = model(z_fixed).detach().cpu()

            final_image = np.clip(final_tensor.squeeze().numpy(), 0.0, 1.0).astype(np.float32)
            duration = round(time.time() - start_time, 2)

            return TrainingResult(
                restored_image=final_image,
                best_iteration=best_iteration,
                loss_history=loss_history,
                psnr_history=psnr_history,
                ssim_history=ssim_history,
                duration_seconds=duration,
                stopped_early=stopped_early,
                cancelled=cancelled,
            )

        finally:
            # Free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
