import os
import sys
import tempfile
import base64
import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import numpy as np
import torch

# Ensure ml package is in python search path
REPO_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = REPO_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from dip_core.config import DIPConfig
from dip_core.pipeline.trainer import train_dip
from dip_core.pipeline.preprocess import load_image
from dip_core.pipeline.postprocess import tensor_to_numpy, apply_thermal_colormap

app = FastAPI(
    title="Thermal DIP Backend",
    description="Minimal FastAPI wrapper for Deep Image Prior thermal restoration.",
    version="1.0.0",
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_DIR = REPO_ROOT / "storage" / "sample_images"


def numpy_to_base64_png(arr: np.ndarray, colormap: Optional[str] = None) -> str:
    """Convert float32 numpy image [0, 1] to base64 PNG data URL."""
    if colormap and colormap != "grayscale":
        rgb = apply_thermal_colormap(arr, colormap)
        pil_img = Image.fromarray(rgb)
    else:
        u8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(u8)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


@app.get("/health")
def health_check():
    """Health check endpoint reporting device and system state."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    return {
        "status": "healthy",
        "device": "cuda" if cuda_available else "cpu",
        "device_name": device_name,
        "torch_version": torch.__version__,
    }


@app.post("/restore")
async def restore_image(
    image: UploadFile = File(..., description="Degraded thermal image"),
    mask: Optional[UploadFile] = File(None, description="Optional defect mask for inpainting"),
    iters: int = Form(1500, description="DIP iterations"),
    task: str = Form("denoise", description="Restoration task: denoise or inpaint"),
    model: str = Form("unet", description="Model architecture: unet or conv"),
):
    """
    Restore degraded thermal image using Deep Image Prior.
    Operates statelessly in a temporary directory and returns base64 PNG results.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Save uploaded image to temp directory
        image_path = tmp_path / f"input_{image.filename}"
        with open(image_path, "wb") as f:
            f.write(await image.read())

        mask_path = None
        if mask:
            mask_path = tmp_path / f"mask_{mask.filename}"
            with open(mask_path, "wb") as f:
                f.write(await mask.read())

        # 2. Configure DIP optimization
        model_type = "conv_baseline" if model == "conv" else "unet"
        config = DIPConfig(
            iterations=iters,
            learning_rate=0.01,
            model_type=model_type,
            device="cuda" if torch.cuda.is_available() else "cpu",
            early_stopping=False,
            run_baselines=True,
        )

        try:
            # 3. Run DIP optimization
            result = train_dip(
                image_input=str(image_path),
                mask_input=str(mask_path) if mask_path else None,
                config=config,
                task=task,
            )

            # 4. Encode images to base64
            orig_np = load_image(str(image_path), grayscale=True)
            orig_b64 = numpy_to_base64_png(orig_np)
            orig_ironbow_b64 = numpy_to_base64_png(orig_np, colormap="ironbow")

            restored_np = result["restored_image"]
            restored_b64 = numpy_to_base64_png(restored_np)
            restored_ironbow_b64 = numpy_to_base64_png(restored_np, colormap="ironbow")

            # Baselines base64 previews
            baselines_data = {}
            for b_name, b_info in result.get("baselines", {}).items():
                baselines_data[b_name] = {
                    "image": numpy_to_base64_png(b_info["image"]),
                    "psnr": b_info.get("psnr"),
                    "ssim": b_info.get("ssim"),
                }

            return {
                "status": "success",
                "task": task,
                "model": model,
                "iterations": iters,
                "original_image": orig_b64,
                "original_ironbow": orig_ironbow_b64,
                "restored_image": restored_b64,
                "restored_ironbow": restored_ironbow_b64,
                "metrics": result["metrics"],
                "baselines": baselines_data,
                "loss_history": result["loss_history"],
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DIP Restoration failed: {str(e)}")


@app.get("/samples")
def get_sample_images():
    """List sample thermal images available in the repository."""
    if not SAMPLE_DIR.exists():
        return []
    samples = []
    for p in SAMPLE_DIR.glob("*.png"):
        samples.append({
            "name": p.stem.replace("_", " ").title(),
            "filename": p.name,
            "url": f"/sample/{p.name}",
        })
    return samples


@app.get("/sample/{filename}")
def get_sample_file(filename: str):
    """Retrieve sample thermal image file."""
    fpath = SAMPLE_DIR / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(str(fpath), media_type="image/png")
