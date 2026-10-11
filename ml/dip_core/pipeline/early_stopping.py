import copy
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


class EarlyStoppingPlateau:
    """
    Monitors training loss for plateaus (since ground-truth PSNR is unavailable at inference).
    Keeps a checkpoint of the best model weights.
    """

    def __init__(
        self,
        patience: int = 300,
        min_delta: float = 1e-5,
        min_iterations: int = 300,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.min_iterations = min_iterations
        self.mode = mode

        self.best_loss = float("inf") if mode == "min" else float("-inf")
        self.best_iteration = 0
        self.best_state_dict: Optional[Dict[str, Any]] = None
        self.best_output: Optional[torch.Tensor] = None
        self.counter = 0
        self.should_stop = False

    def step(
        self,
        loss: float,
        iteration: int,
        model: nn.Module,
        current_output: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        Update with current step's loss.
        Returns True if early stopping criteria triggered.
        """
        is_better = False
        if self.mode == "min":
            if (self.best_loss - loss) > self.min_delta:
                is_better = True
        else:
            if (loss - self.best_loss) > self.min_delta:
                is_better = True

        if is_better or self.best_state_dict is None:
            self.best_loss = loss
            self.best_iteration = iteration
            self.counter = 0
            # Save lightweight CPU state dict copy
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if current_output is not None:
                self.best_output = current_output.detach().cpu().clone()
        else:
            self.counter += 1
            if iteration >= self.min_iterations and self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def restore_best_model(self, model: nn.Module) -> None:
        """Restore model to best saved state dict."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
