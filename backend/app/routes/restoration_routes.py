"""
Restoration job endpoints: creation, status polling, result retrieval, cancellation, download.
"""
import uuid
import json
import threading
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response
from backend.app.schemas import (
    JobCreateResponse,
    JobStatusResponse,
    JobResultResponse,
    JobStatusEnum,
)
from backend.app.services.job_store import job_store
from backend.app.services.image_io import load_image_from_bytes, load_mask_from_bytes
from backend.app.services.restoration_service import execute_restoration_job

router = APIRouter(prefix="/restorations", tags=["Restoration"])

@router.post("", response_model=JobCreateResponse)
async def create_restoration_job(
    image: UploadFile = File(..., description="Thermal image to restore (PNG/JPEG)"),
    mask: Optional[UploadFile] = File(None, description="Optional binary mask image"),
    options: Optional[str] = Form(None, description="JSON string of RestorationRequestOptions")
):
    """
    Creates a new restoration job and begins background GPU optimization.
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")
        
    try:
        image_arr = load_image_from_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")
        
    mask_arr = None
    if mask is not None:
        mask_bytes = await mask.read()
        if mask_bytes:
            try:
                mask_arr = load_mask_from_bytes(mask_bytes, image_arr.shape)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode mask: {str(e)}")

    opts = {}
    if options:
        try:
            opts = json.loads(options)
        except Exception:
            opts = {}

    job_id = f"dip_{uuid.uuid4().hex[:10]}"
    job_store.create_job(job_id, opts)

    # Launch worker in background thread
    thread = threading.Thread(
        target=execute_restoration_job,
        args=(job_id, image_arr, mask_arr, opts),
        daemon=True
    )
    thread.start()

    return JobCreateResponse(
        job_id=job_id,
        status=JobStatusEnum.PENDING,
        message="Restoration job successfully created and scheduled for processing."
    )

@router.get("/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """
    Poll the current status, iteration count, current loss, and intermediate preview.
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
        
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress_percentage=job["progress_percentage"],
        current_iteration=job["current_iteration"],
        total_iterations=job["total_iterations"],
        current_loss=job["current_loss"],
        preview_image_base64=job["preview_image_base64"],
        degraded_image_base64=job.get("degraded_image_base64"),
        eta_seconds=job["eta_seconds"],
        error_message=job["error_message"]
    )

@router.get("/{job_id}/result", response_model=JobResultResponse)
def get_job_result(job_id: str):
    """
    Fetch the completed restoration results, metrics, losses array, and high-res images.
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
        
    return JobResultResponse(
        job_id=job["job_id"],
        status=job["status"],
        restored_image_base64=job["restored_image_base64"],
        original_image_base64=job["original_image_base64"],
        degraded_image_base64=job["degraded_image_base64"],
        mask_image_base64=job["mask_image_base64"],
        metrics=job["metrics"],
        losses=job["losses"],
        runtime_seconds=job["runtime_seconds"],
        error_message=job["error_message"]
    )

@router.post("/{job_id}/cancel")
def cancel_job(job_id: str):
    """
    Cancels an active restoration optimization run.
    """
    success = job_store.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or already terminated.")
    return {"status": "cancelled", "job_id": job_id}

@router.get("/{job_id}/download")
def download_restored_image(job_id: str):
    """
    Download the restored thermal image as a raw PNG file.
    """
    import base64
    job = job_store.get_job(job_id)
    if not job or not job.get("restored_image_base64"):
        raise HTTPException(status_code=404, detail="Restored image not available.")
        
    data_uri = job["restored_image_base64"]
    header, encoded = data_uri.split(",", 1)
    file_bytes = base64.b64decode(encoded)
    
    return Response(
        content=file_bytes,
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename=restored_{job_id}.png"}
    )
