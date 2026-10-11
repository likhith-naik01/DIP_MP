from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DIPConfig:
    """Hyperparameters and configuration for Deep Image Prior training."""

    # Optimization
    iterations: int = 1500
    learning_rate: float = 0.01
    optimizer: str = "adam"
    weight_decay: float = 0.0

    # Input Noise
    input_channels: int = 32
    noise_std: float = 0.1  # initial noise tensor std
    reg_noise_std: float = 1.0 / 30.0  # additive noise std per iteration as regularizer (1/30)

    # Model architecture
    model_type: str = "unet"  # "unet" or "conv_baseline"
    encoder_channels: Tuple[int, ...] = (16, 32, 64, 128, 128)
    skip_channels: int = 4  # skip projection dimension
    out_channels: int = 1  # 1 for thermal grayscale

    # Early stopping & Plateau detection
    early_stopping: bool = False  # optional flag
    patience: int = 250
    min_delta: float = 1e-5

    # Progress reporting
    log_interval: int = 50

    # Device
    device: str = "cuda"

    # Baselines
    run_baselines: bool = True
