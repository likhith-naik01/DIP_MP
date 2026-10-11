import sys
from pathlib import Path

# Add ml folder and root to python path so dip_core is importable
repo_root = Path(__file__).resolve().parent.parent.parent
ml_dir = repo_root / "ml"
for p in [repo_root, ml_dir]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
