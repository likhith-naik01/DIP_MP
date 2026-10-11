"""Configuration dataclasses and device helpers for Deep Image Prior thermal restoration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union, Any, Tuple
import torch


class RestorationTask(str, Enum):
    """Supported thermal image restoration tasks."""
    AUTO = "auto"
    DENOISE = "denoise"
    INPAINT = "inpaint"
    DENOISE_AND_INPAINT = "denoise_and_inpaint"
    DENOISING = "denoising"
    INPAINTING = "inpainting"
    SUPER_RESOLUTION = "super_resolution"
    COMBINED = "combined"


@dataclass
class DipTrainingConfig:
    """Hyperparameters for Deep Image Prior network optimization."""
    iterations: int = 1500
    learning_rate: float = 0.01
    noise_channels: int = 32
    reg_noise_std: float = 1.0 / 30.0  # Gaussian jitter added to z each iteration
    noise_std: float = 0.1             # Initial noise tensor standard deviation
    log_every: int = 50                # Logging and preview frequency
    save_every: int = 50               # Checkpoint/snapshot save frequency
    model_type: str = "unet"           # "unet" or "conv_baseline"
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 250
    early_stopping_min_delta: float = 1e-5
    smoothing_window: int = 20
    seed: int = 42
    out_channels: int = 1              # Output image channels
    skip_channels: int = 4             # 1x1 skip connection channels
    weight_decay: float = 0.0          # Adam optimizer weight decay
    run_baselines: bool = True         # Whether to compute classical baselines
    device: Optional[str] = None       # "cuda", "cpu", or None for auto-detect
    input_channels: Optional[int] = None
    log_interval: Optional[int] = None
    early_stopping: Optional[bool] = None
    patience: Optional[int] = None
    min_delta: Optional[float] = None

    def __post_init__(self):
        if self.input_channels is not None:
            self.noise_channels = self.input_channels
        else:
            self.input_channels = self.noise_channels
        if self.log_interval is not None:
            self.log_every = self.log_interval
        else:
            self.log_interval = self.log_every
        if self.early_stopping is not None:
            self.early_stopping_enabled = self.early_stopping
        else:
            self.early_stopping = self.early_stopping_enabled
        if self.patience is not None:
            self.early_stopping_patience = self.patience
        else:
            self.patience = self.early_stopping_patience
        if self.min_delta is not None:
            self.early_stopping_min_delta = self.min_delta
        else:
            self.min_delta = self.early_stopping_min_delta


# Backward-compatibility alias
DIPConfig = DipTrainingConfig


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
    # Flexible aliases
    dead_pixel_ratio: Optional[float] = None
    scratch_line_count: Optional[int] = None
    scratch_line_thickness: Optional[int] = None
    rect_hole_count: Optional[int] = None
    rect_hole_size: Optional[Any] = None

    def __post_init__(self):
        if self.dead_pixel_ratio is not None:
            self.dropout_ratio = self.dead_pixel_ratio
        else:
            self.dead_pixel_ratio = self.dropout_ratio
        if self.scratch_line_count is not None:
            self.num_line_holes = self.scratch_line_count
        else:
            self.scratch_line_count = self.num_line_holes
        if self.rect_hole_count is not None:
            self.num_rect_holes = self.rect_hole_count
        else:
            self.rect_hole_count = self.num_rect_holes


def get_device(requested: Optional[str] = None) -> torch.device:
    """
    Return torch.device: selects 'cuda' if available, otherwise falls back to 'cpu'.
    """
    if requested and requested.lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
