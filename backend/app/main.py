"""
Main FastAPI entrypoint for the Deep Image Prior Thermal Restoration API.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

    @app.get("/")
    def root():
        return {
            "name": "Thermal Image Restoration API (Deep Image Prior)",
            "status": "online",
            "compute_device": DEVICE,
            "gpu_hardware": GPU_NAME,
            "docs": "/docs",
            "health": "/api/health"
        }

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
