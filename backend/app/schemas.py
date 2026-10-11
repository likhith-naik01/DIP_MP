"""
Pydantic schemas for request/response serialization.
"""
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class RestorationTaskEnum(str, Enum):
    INPAINTING = "inpainting"
    DENOISING = "denoising"
    SUPER_RESOLUTION = "super_resolution"
    COMBINED = "combined"

class ModelNameEnum(str, Enum):
    UNET = "unet"
    CONV_BASELINE = "conv_baseline"

class ColormapEnum(str, Enum):
    GRAYSCALE = "grayscale"
    IRONBOW = "ironbow"
    INFERNO = "inferno"
    VIRIDIS = "viridis"

class JobStatusEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class RestorationRequestOptions(BaseModel):
    task: RestorationTaskEnum = RestorationTaskEnum.INPAINTING
    model: ModelNameEnum = ModelNameEnum.UNET
    iterations: int = Field(default=1500, ge=10, le=5000)
    learning_rate: float = Field(default=0.01, gt=0.0, le=0.1)
    reg_noise_std: float = Field(default=0.03, ge=0.0, le=0.5)
    colormap: ColormapEnum = ColormapEnum.IRONBOW
    auto_detect_mask: bool = False
    simulate_damage: bool = False
    early_stopping: bool = True
    patience: int = 250

class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatusEnum
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatusEnum
    progress_percentage: float = 0.0
    current_iteration: int = 0
    total_iterations: int = 0
    current_loss: Optional[float] = None
    preview_image_base64: Optional[str] = None
    eta_seconds: Optional[float] = None
    error_message: Optional[str] = None

class JobResultResponse(BaseModel):
    job_id: str
    status: JobStatusEnum
    restored_image_base64: Optional[str] = None
    original_image_base64: Optional[str] = None
    degraded_image_base64: Optional[str] = None
    mask_image_base64: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    losses: List[float] = Field(default_factory=list)
    runtime_seconds: float = 0.0
    error_message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    cuda_available: bool
    gpu_name: str
