"""
Configuration settings for the DIP Backend Application.
"""
from pathlib import Path
import torch

# Base paths (Using only Pathlib and relative/environment paths)
BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Constraints
MAX_UPLOAD_SIZE_MB: int = 10
MAX_IMAGE_DIM: int = 512
ALLOWED_IMAGE_TYPES: set = {"image/png", "image/jpeg", "image/jpg", "image/tiff", "image/bmp"}

# Device configuration
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME: str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)"
