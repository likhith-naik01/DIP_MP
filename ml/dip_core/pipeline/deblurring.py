"""
Physics-informed deblurring, Richardson-Lucy deconvolution, and multi-scale detail restoration.
Designed specifically for thermal infrared and digital sensor images suffering from blurriness,
diffraction, low spatial frequency response, and optical/motion degradation.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2


def compute_sharpness_metrics(image: np.ndarray) -> Dict[str, Any]:
    """
    Computes rigorous blur and sharpness metrics on a normalized float32 [0, 1] image:
      - Laplacian variance: standard edge variance. Higher = sharper.
      - Tenengrad energy: sum of squared Sobel gradient magnitudes.
      - High-frequency spectral energy: ratio of high frequencies via FFT.
      - is_blurry: boolean flag determined by thresholding.
    """
    img_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    
    # 1. Laplacian variance
    lap = cv2.Laplacian(img_u8, cv2.CV_64F)
    lap_var = float(lap.var())
    
    # 2. Tenengrad gradient energy
    sobel_x = cv2.Sobel(img_u8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_u8, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(sobel_x ** 2 + sobel_y ** 2))
    
    # 3. Frequency domain high-pass energy ratio
    h, w = img_u8.shape[:2]
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.abs(dft_shift)
    
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    y, x = np.ogrid[:h, :w]
    mask_low = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    
    total_energy = float(np.sum(magnitude_spectrum)) + 1e-8
    low_energy = float(np.sum(magnitude_spectrum[mask_low]))
    high_energy_ratio = float((total_energy - low_energy) / total_energy)
    
    # Blurriness determination: Laplacian variance < 15.0 or Tenengrad < 2200 in 8-bit scale
    is_blurry = lap_var < 15.0 or tenengrad < 2200.0 or high_energy_ratio < 0.45
    
    return {
        "laplacian_variance": round(lap_var, 3),
        "tenengrad_energy": round(tenengrad, 2),
        "high_energy_ratio": round(high_energy_ratio, 4),
        "is_blurry": is_blurry,
    }


def estimate_blur_psf(
    kernel_size: int = 5,
    sigma: float = 1.4,
    angle: Optional[float] = None
) -> np.ndarray:
    """
    Creates an optical Point Spread Function (PSF) kernel.
    Defaults to isotropic 2D Gaussian PSF modeling microbolometer / camera lens diffraction.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    if angle is None:
        # Isotropic Gaussian PSF
        k1d = cv2.getGaussianKernel(kernel_size, sigma)
        psf = k1d @ k1d.T
    else:
        # Directional motion blur kernel
        psf = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        for i in range(-center, center + 1):
            x = int(round(center + i * dx))
            y = int(round(center + i * dy))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                psf[y, x] = 1.0
        if psf.sum() == 0:
            psf[center, center] = 1.0
            
    psf /= psf.sum()
    return psf.astype(np.float32)


def richardson_lucy_deconvolution(
    image: np.ndarray,
    psf: np.ndarray,
    iterations: int = 16,
    damping: float = 0.001,
) -> np.ndarray:
    """
    Poisson Maximum-Likelihood Richardson-Lucy deconvolution.
    Inverts optical blur while strictly preserving physical thermal radiometry [0.0, 1.0].
    Includes border reflection tapering to eliminate boundary ringing.
    """
    img = np.clip(image, 0.0, 1.0).astype(np.float32)
    h, w = img.shape[:2]
    
    # Pad borders to prevent boundary edge reflection ringing
    pad_len = max(psf.shape) * 2
    padded = cv2.copyMakeBorder(img, pad_len, pad_len, pad_len, pad_len, cv2.BORDER_REFLECT)
    
    psf_flipped = psf[::-1, ::-1].astype(np.float32)
    latent = np.maximum(padded, 1e-6)
    
    for _ in range(iterations):
        # Forward blur prediction: k * x
        reconv = cv2.filter2D(latent, -1, psf)
        reconv = np.maximum(reconv, 1e-6)
        
        # Relative blur ratio: y / (k * x)
        ratio = padded / reconv
        
        if damping > 0.0:
            # Damped correction to prevent noise amplification in flat regions
            diff = ratio - 1.0
            damped_ratio = 1.0 + np.sign(diff) * np.maximum(0.0, np.abs(diff) - damping)
            correction = cv2.filter2D(damped_ratio, -1, psf_flipped)
        else:
            correction = cv2.filter2D(ratio, -1, psf_flipped)
            
        latent = np.clip(latent * correction, 0.0, 1.0)
        
    # Crop back unpadded region
    unpadded = latent[pad_len : pad_len + h, pad_len : pad_len + w]
    return np.clip(unpadded, 0.0, 1.0).astype(np.float32)


def wiener_deconvolution(
    image: np.ndarray,
    psf: np.ndarray,
    nsr: float = 0.02
) -> np.ndarray:
    """
    Fast frequency-domain Wiener filter deconvolution.
    Useful for rapid sharp prior estimation.
    """
    img = np.clip(image, 0.0, 1.0).astype(np.float32)
    h, w = img.shape[:2]
    
    # Pad PSF to image shape
    psf_padded = np.zeros((h, w), dtype=np.float32)
    kh, kw = psf.shape
    psf_padded[:kh, :kw] = psf
    psf_padded = np.roll(psf_padded, -kh // 2, axis=0)
    psf_padded = np.roll(psf_padded, -kw // 2, axis=1)
    
    # FFT
    img_fft = np.fft.fft2(img)
    psf_fft = np.fft.fft2(psf_padded)
    
    # Wiener transfer function: H* / (|H|^2 + NSR)
    psf_conj = np.conj(psf_fft)
    psf_mag_sq = np.abs(psf_fft) ** 2
    transfer = psf_conj / (psf_mag_sq + nsr)
    
    deconv_fft = img_fft * transfer
    deconv = np.fft.ifft2(deconv_fft)
    
    return np.clip(np.real(deconv), 0.0, 1.0).astype(np.float32)


def multi_scale_detail_recovery(
    base_image: np.ndarray,
    sharp_prior: np.ndarray,
    detail_weight: float = 1.25,
    coring_threshold: float = 0.015,
) -> np.ndarray:
    """
    Extracts high-frequency textural details and gradient edges from the sharp prior
    and fuses them with the base representation using edge-coring noise thresholding.
    Prevents sensor noise amplification in flat regions while making genuine thermal edges razor-sharp.
    """
    # High-pass filter sharp prior via Gaussian difference (Laplacian pyramid layer)
    smooth = cv2.GaussianBlur(sharp_prior, (0, 0), sigmaX=1.1)
    high_pass = sharp_prior - smooth
    
    # Edge coring: only boost details whose magnitude exceeds coring threshold
    edge_mask = np.abs(high_pass) > coring_threshold
    boosted_detail = high_pass * edge_mask * detail_weight
    
    fused = base_image + boosted_detail
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def adaptive_unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.2,
    strength: float = 1.3,
    threshold: float = 0.012,
) -> np.ndarray:
    """
    Anti-halo adaptive unsharp masking with edge-coring threshold.
    Enhances micro-contrast across thermal boundaries without creating black/white halo artifacts.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    diff = image - blurred
    
    # Apply soft-thresholding coring
    cored_diff = np.sign(diff) * np.maximum(0.0, np.abs(diff) - threshold)
    sharpened = image + (strength * cored_diff)
    
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)


def thermal_clahe_enhancement(
    image: np.ndarray,
    clip_limit: float = 1.6,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Contrast-Limited Adaptive Histogram Equalization calibrated for infrared thermal images.
    Expands compressed local temperature gradients without over-amplifying background noise.
    """
    u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(u8)
    return (enhanced.astype(np.float32) / 255.0)


def strong_deblur_pipeline(
    image: np.ndarray,
    strength: str = "strong",
    psf_size: int = 5,
    psf_sigma: float = 1.4,
    iterations: int = 16,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    End-to-end Strong Deblurring & Super-Resolution Enhancement Pipeline:
      1. Blur estimation & characterization.
      2. Point Spread Function (PSF) model construction.
      3. Richardson-Lucy iterative deconvolution with edge-preservation.
      4. Multi-scale detail recovery with anti-noise coring.
      5. Adaptive anti-halo unsharp masking.
      6. Gentle thermal local contrast enhancement (CLAHE).
      7. Dynamic range preservation.
      
    Returns:
      (restored_image, metrics_dict)
    """
    initial_metrics = compute_sharpness_metrics(image)
    
    # Adjust parameters based on strength setting
    if strength == "ultra":
        psf_sigma_adj = psf_sigma * 1.2
        rl_iters = max(iterations, 20)
        detail_weight = 1.5
        unsharp_strength = 1.6
        clahe_clip = 1.8
        coring_thresh = 0.010
    elif strength == "normal":
        psf_sigma_adj = psf_sigma * 0.9
        rl_iters = min(iterations, 12)
        detail_weight = 1.0
        unsharp_strength = 1.1
        clahe_clip = 1.4
        coring_thresh = 0.018
    else:  # "strong" default
        psf_sigma_adj = psf_sigma
        rl_iters = iterations
        detail_weight = 1.3
        unsharp_strength = 1.35
        clahe_clip = 1.6
        coring_thresh = 0.014
        
    # 1. Estimate PSF
    psf = estimate_blur_psf(kernel_size=psf_size, sigma=psf_sigma_adj)
    
    # 2. Richardson-Lucy Deconvolution
    deconv_rl = richardson_lucy_deconvolution(
        image=image,
        psf=psf,
        iterations=rl_iters,
        damping=0.001
    )
    
    # 3. Multi-scale Detail Extraction and High-Frequency Injection
    detail_fused = multi_scale_detail_recovery(
        base_image=image,
        sharp_prior=deconv_rl,
        detail_weight=detail_weight,
        coring_threshold=coring_thresh,
    )
    
    # 4. Adaptive Unsharp Masking
    unsharp = adaptive_unsharp_mask(
        image=detail_fused,
        sigma=1.1,
        strength=unsharp_strength,
        threshold=coring_thresh,
    )
    
    # 5. Local Contrast Tone Mapping (CLAHE)
    clahe_img = thermal_clahe_enhancement(
        image=unsharp,
        clip_limit=clahe_clip,
        tile_grid_size=(8, 8)
    )
    
    # 6. Radiometrically-sound blend: 70% sharp details + 30% CLAHE dynamic expansion
    final_restored = np.clip(0.70 * unsharp + 0.30 * clahe_img, 0.0, 1.0).astype(np.float32)
    
    # 7. Final sharpness measurement
    final_metrics = compute_sharpness_metrics(final_restored)
    
    orig_lap = max(initial_metrics["laplacian_variance"], 1e-4)
    final_lap = final_metrics["laplacian_variance"]
    sharpness_gain_pct = round(((final_lap / orig_lap) - 1.0) * 100.0, 1)
    
    orig_ten = max(initial_metrics["tenengrad_energy"], 1e-4)
    final_ten = final_metrics["tenengrad_energy"]
    tenengrad_gain_pct = round(((final_ten / orig_ten) - 1.0) * 100.0, 1)
    
    pipeline_metrics = {
        "sharpness_before": initial_metrics["laplacian_variance"],
        "sharpness_after": final_metrics["laplacian_variance"],
        "sharpness_gain_pct": sharpness_gain_pct,
        "tenengrad_before": initial_metrics["tenengrad_energy"],
        "tenengrad_after": final_metrics["tenengrad_energy"],
        "tenengrad_gain_pct": tenengrad_gain_pct,
        "is_blurry_before": initial_metrics["is_blurry"],
        "is_blurry_after": final_metrics["is_blurry"],
        "deblur_strength": strength,
    }
    
    return final_restored, pipeline_metrics
