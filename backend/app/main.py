import sys
from pathlib import Path

# Ensure ML package and repo root are in Python search path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ML_DIR = REPO_ROOT / "ml"
for p in [str(ML_DIR), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.app.routes.health_routes import router as health_router
from backend.app.routes.restoration_routes import router as restoration_router
from backend.app.config import DEVICE, GPU_NAME

def create_app() -> FastAPI:
    app = FastAPI(
        title="Thermal Image Restoration API (Deep Image Prior)",
        description=(
            "Unsupervised single-image thermal restoration using a 5-level U-Net generator. "
            "Optimizes directly on the degraded input image without pre-training datasets."
        ),
        version="1.0.0"
    )

    # Enable full Cross-Origin Resource Sharing for the web client
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routers
    app.include_router(health_router, prefix="/api")
    app.include_router(restoration_router, prefix="/api")

    # Mount Frontend UI directly so the app can be used in 1 click
    frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
