"""Pipeline modules for Deep Image Prior thermal restoration."""

from dip_core.pipeline.image_preprocessing import (
    load_image,
    pad_to_multiple_of_32,
    crop_back_to_original_size,
    to_tensor,
    to_device,
    build_valid_mask,
)
from dip_core.pipeline.mask_detection import detect_damage_mask
from dip_core.pipeline.damage_simulation import simulate_damage
from dip_core.pipeline.noise_input import create_fixed_noise, perturb_noise
from dip_core.pipeline.masked_loss import MaskedMSELoss, masked_mse_loss
from dip_core.pipeline.early_stopping import EarlyStopper
from dip_core.pipeline.dip_trainer import DipTrainer, TrainingResult
from dip_core.pipeline.image_postprocessing import (
    composite_restored_image,
    apply_thermal_colormap,
    save_image,
    to_uint8,
)
from dip_core.pipeline.restoration_pipeline import run_restoration, RestorationOutput

__all__ = [
    "load_image",
    "pad_to_multiple_of_32",
    "crop_back_to_original_size",
    "to_tensor",
    "to_device",
    "build_valid_mask",
    "detect_damage_mask",
    "simulate_damage",
    "create_fixed_noise",
    "perturb_noise",
    "MaskedMSELoss",
    "masked_mse_loss",
    "EarlyStopper",
    "DipTrainer",
    "TrainingResult",
    "composite_restored_image",
    "apply_thermal_colormap",
    "save_image",
    "to_uint8",
    "run_restoration",
    "RestorationOutput",
]
