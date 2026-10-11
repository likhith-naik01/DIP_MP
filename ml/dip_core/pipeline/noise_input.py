from typing import Tuple, Union
import torch


def get_noise_input(
    spatial_shape: Tuple[int, int],
    channels: int = 32,
    device: Union[str, torch.device] = "cpu",
    std: float = 0.1,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate fixed input noise tensor z ~ N(0, std^2 * I).
    Shape: (1, channels, H, W).
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    h, w = spatial_shape
    # Draw from standard normal on CPU with seed, then transfer to device
    noise = torch.randn((1, channels, h, w), generator=g, dtype=torch.float32) * std
    return noise.to(device)


def perturb_noise(
    z: torch.Tensor,
    reg_noise_std: float = 0.03,
) -> torch.Tensor:
    """
    Add small stochastic jitter to z at each training step as a regularizer
    (standard DIP regularization to prevent overfitting to high-frequency noise).
    """
    if reg_noise_std <= 0:
        return z
    jitter = torch.randn_like(z) * reg_noise_std
    return z + jitter
