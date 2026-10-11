"""
Root entrypoint redirecting to backend.app.main:app for backwards compatibility.
"""
import sys
from pathlib import Path

# Ensure ML package and backend package are importable
REPO_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = REPO_ROOT / "ml"
BACKEND_DIR = REPO_ROOT / "backend"

for p in [REPO_ROOT, ML_DIR, BACKEND_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backend.app.main import app, create_app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
