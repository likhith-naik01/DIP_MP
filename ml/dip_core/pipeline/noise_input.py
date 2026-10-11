"""Noise input generator and regularization perturbation for Deep Image Prior."""

from typing import Tuple, Optional, Union
import torch
from dip_core.config import get_device


def create_fixed_noise(
    spatial_shape: Tuple[int, int],
    channels: int = 32,
    std: float = 0.1,
    seed: int = 42,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> torch.Tensor:
    """
    Generate the fixed input noise tensor z ~ N(0, std^2 * I).
    Shape: (1, channels, H, W).
    """
    dev = get_device(device if isinstance(device, str) else None) if (device is None or isinstance(device, str)) else device
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    h, w = spatial_shape
    noise = torch.randn((1, channels, h, w), generator=g, dtype=torch.float32) * std
    return noise.to(dev)


def perturb_noise(
    z: torch.Tensor,
    reg_noise_std: float = 1.0 / 30.0,
) -> torch.Tensor:
    """
    Perturb fixed noise tensor z with small stochastic Gaussian jitter each step.
    This acts as a regularizer to prevent overfitting to high-frequency noise.
    """
    if reg_noise_std <= 0:
        return z
    jitter = torch.randn_like(z) * reg_noise_std
    return z + jitter


# Backward compatibility aliases
get_noise_input = create_fixed_noise
