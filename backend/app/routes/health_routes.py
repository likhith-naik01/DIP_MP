"""
Health and diagnostics routes.
"""
from fastapi import APIRouter
from backend.app.schemas import HealthResponse
from backend.app.config import DEVICE, GPU_NAME
import torch

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
def get_health():
    """
    Returns API server health, compute device, and GPU status.
    """
    return HealthResponse(
        status="ok",
        version="1.0.0",
        device=DEVICE,
        cuda_available=torch.cuda.is_available(),
        gpu_name=GPU_NAME
    )
