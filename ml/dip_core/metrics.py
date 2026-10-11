import math
from typing import Union, Optional
import numpy as np
import torch
import torch.nn.functional as F


def calculate_psnr(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    max_val: float = 1.0,
    eps: float = 1e-10,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images in [0, 1].
    If mask is provided, computes PSNR only where mask == 1.
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().squeeze().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().squeeze().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().squeeze().numpy()

    if mask is not None:
        valid_idx = mask > 0.5
        if not np.any(valid_idx):
            return 0.0
        mse = float(np.mean((img1[valid_idx] - img2[valid_idx]) ** 2))
    else:
        mse = float(np.mean((img1 - img2) ** 2))

    if mse <= eps:
        return 100.0
    return float(10.0 * math.log10((max_val**2) / mse))


def calculate_ssim(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images normalized to [0, 1].
    Uses skimage.metrics.structural_similarity when available, with PyTorch fallback.
    """
    if isinstance(img1, torch.Tensor):
        arr1 = img1.detach().cpu().squeeze().numpy()
    else:
        arr1 = np.squeeze(img1)

    if isinstance(img2, torch.Tensor):
        arr2 = img2.detach().cpu().squeeze().numpy()
    else:
        arr2 = np.squeeze(img2)

    if mask is not None:
        if isinstance(mask, torch.Tensor):
            m_arr = mask.detach().cpu().squeeze().numpy()
        else:
            m_arr = np.squeeze(mask)
    else:
        m_arr = None

    # Try skimage first
    try:
        from skimage.metrics import structural_similarity as ssim_fn

        if m_arr is not None:
            # Full ssim map with gradient
            score, ssim_map = ssim_fn(arr1, arr2, data_range=1.0, full=True)
            valid = m_arr > 0.5
            if np.any(valid):
                return float(np.mean(ssim_map[valid]))
            return float(score)
        else:
            return float(ssim_fn(arr1, arr2, data_range=1.0))
    except Exception:
        pass

    # Native PyTorch fallback
    t1 = torch.from_numpy(arr1).float().unsqueeze(0).unsqueeze(0)
    t2 = torch.from_numpy(arr2).float().unsqueeze(0).unsqueeze(0)

    # 11x11 Gaussian window
    window_size = 11
    gauss = torch.tensor(
        [math.exp(-((x - window_size // 2) ** 2) / (2 * 1.5**2)) for x in range(window_size)],
        dtype=torch.float32,
    )
    gauss = gauss / gauss.sum()
    _1d = gauss.unsqueeze(1)
    window = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)

    mu1 = F.conv2d(t1, window, padding=window_size // 2)
    mu2 = F.conv2d(t2, window, padding=window_size // 2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(t1 * t1, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(t2 * t2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(t1 * t2, window, padding=window_size // 2) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if m_arr is not None:
        m_t = torch.from_numpy(m_arr).float().unsqueeze(0).unsqueeze(0)
        valid_count = torch.sum(m_t)
        if valid_count > 0:
            return float((torch.sum(ssim_map * m_t) / valid_count).item())

    return float(ssim_map.mean().item())
