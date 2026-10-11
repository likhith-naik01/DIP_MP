"""Early stopping and moving-average loss plateau detection for Deep Image Prior."""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn


class EarlyStopper:
    """
    Monitors moving-average training loss for convergence plateaus.

    NOTE ON HEURISTIC NATURE:
    In real-world deployment without ground-truth reference images, optimal stopping
    cannot use PSNR or SSIM. DIP networks fit natural image structures before fitting
    high-frequency noise (spectral bias). Selecting the best iteration using smoothed
    loss plateau or a bounded iteration limit is an empirical heuristic to prevent
    late-stage noise memorization.
    """

    def __init__(
        self,
        patience: int = 250,
        min_delta: float = 1e-5,
        smoothing_window: int = 20,
        min_iterations: int = 300,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_window = smoothing_window
        self.min_iterations = min_iterations

        self.smoothed_loss: Optional[float] = None
        self.best_loss = float("inf")
        self.best_iteration = 0
        self.best_output: Optional[torch.Tensor] = None
        self.counter = 0
        self.should_stop = False
        self._alpha = 2.0 / (smoothing_window + 1.0)

    def step(
        self,
        loss: float,
        iteration: int,
        current_output: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        Update tracker with current iteration's loss.
        Returns True if early stopping plateau is triggered.
        """
        # Exponential Moving Average update
        if self.smoothed_loss is None:
            self.smoothed_loss = loss
        else:
            self.smoothed_loss = self._alpha * loss + (1.0 - self._alpha) * self.smoothed_loss

        # Check improvement against best smoothed loss
        if (self.best_loss - self.smoothed_loss) > self.min_delta:
            self.best_loss = self.smoothed_loss
            self.best_iteration = iteration
            self.counter = 0
            if current_output is not None:
                self.best_output = current_output.detach().cpu().clone()
        else:
            self.counter += 1
            if iteration >= self.min_iterations and self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# Backward compatibility alias
EarlyStoppingPlateau = EarlyStopper
