"""Configuration dataclasses and device helpers for Deep Image Prior thermal restoration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union
import torch


class RestorationTask(str, Enum):
    """Supported thermal image restoration tasks."""
    DENOISE = "denoise"
    INPAINT = "inpaint"
    DENOISE_AND_INPAINT = "denoise_and_inpaint"


@dataclass
class DipTrainingConfig:
    """Hyperparameters for Deep Image Prior network optimization."""
    iterations: int = 1500
    learning_rate: float = 0.01
    noise_channels: int = 32
    reg_noise_std: float = 1.0 / 30.0  # Gaussian jitter added to z each iteration
    noise_std: float = 0.1             # Initial noise tensor standard deviation
    log_every: int = 50                # Logging and preview frequency
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 250
    early_stopping_min_delta: float = 1e-5
    smoothing_window: int = 20
    seed: int = 42
    device: Optional[str] = None       # "cuda", "cpu", or None for auto-detect


@dataclass
class ModelConfig:
    """Architecture configuration for the generator network."""
    model_name: str = "unet"           # "unet" or "conv_baseline"
    in_channels: int = 32
    out_channels: int = 1              # Thermal images are single-channel grayscale
    encoder_channels: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 128])
    skip_channels: int = 4             # 1x1 conv projection channel count for skips
    upsample_mode: str = "bilinear"


@dataclass
class DamageConfig:
    """Configuration for synthetic damage simulation on clean thermal images."""
    gaussian_snr_db: Optional[float] = 20.0  # Target SNR in decibels
    gaussian_noise_std: float = 0.10         # Used if gaussian_snr_db is None
    dropout_ratio: float = 0.03              # Random dead/corrupted pixel ratio
    num_line_holes: int = 4                  # Number of scratch lines
    num_rect_holes: int = 2                  # Number of rectangular missing blocks
    seed: int = 42


def get_device(requested: Optional[str] = None) -> torch.device:
    """
    Return torch.device: selects 'cuda' if available, otherwise falls back to 'cpu'.
    """
    if requested and requested.lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
