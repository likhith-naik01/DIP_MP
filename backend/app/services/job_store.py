"""
In-memory job store for restoration tasks.
No database, no auth required. Thread-safe with cancellation events.
"""
import threading
from typing import Dict, Any, Optional
from backend.app.schemas import JobStatusEnum

class JobStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._execution_lock = threading.Lock() # Ensure single GPU task at a time
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._cancel_events: Dict[str, threading.Event] = {}

    def create_job(self, job_id: str, options: dict) -> None:
        with self._lock:
            cancel_ev = threading.Event()
            self._cancel_events[job_id] = cancel_ev
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": JobStatusEnum.PENDING,
                "options": options,
                "progress_percentage": 0.0,
                "current_iteration": 0,
                "total_iterations": options.get("iterations", 1500),
                "current_loss": None,
                "preview_image_base64": None,
                "eta_seconds": None,
                "restored_image_base64": None,
                "original_image_base64": None,
                "degraded_image_base64": None,
                "mask_image_base64": None,
                "metrics": {},
                "losses": [],
                "runtime_seconds": 0.0,
                "error_message": None,
                "cancel_event": cancel_ev,
            }

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_cancel_event(self, job_id: str) -> Optional[threading.Event]:
        with self._lock:
            return self._cancel_events.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._cancel_events:
                self._cancel_events[job_id].set()
                if job_id in self._jobs:
                    self._jobs[job_id]["status"] = JobStatusEnum.CANCELLED
                return True
            return False

    def update_progress(
        self,
        job_id: str,
        current_iter: int,
        total_iter: int,
        loss: float,
        preview_base64: Optional[str] = None,
        eta_seconds: Optional[float] = None
    ) -> None:
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job["status"] = JobStatusEnum.RUNNING
                job["current_iteration"] = current_iter
                job["total_iterations"] = total_iter
                job["progress_percentage"] = round((current_iter / max(total_iter, 1)) * 100.0, 1)
                job["current_loss"] = float(loss)
                if preview_base64 is not None:
                    job["preview_image_base64"] = preview_base64
                if eta_seconds is not None:
                    job["eta_seconds"] = round(eta_seconds, 1)
                job["losses"].append(float(loss))

    def complete_job(
        self,
        job_id: str,
        restored_base64: str,
        original_base64: Optional[str],
        degraded_base64: Optional[str],
        mask_base64: Optional[str],
        metrics: dict,
        losses: list,
        runtime_seconds: float
    ) -> None:
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job["status"] = JobStatusEnum.COMPLETED
                job["progress_percentage"] = 100.0
                job["restored_image_base64"] = restored_base64
                job["original_image_base64"] = original_base64
                job["degraded_image_base64"] = degraded_base64
                job["mask_image_base64"] = mask_base64
                job["metrics"] = metrics
                job["losses"] = losses
                job["runtime_seconds"] = round(runtime_seconds, 2)
                job["preview_image_base64"] = restored_base64

    def fail_job(self, job_id: str, error_message: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job["status"] = JobStatusEnum.FAILED
                job["error_message"] = error_message

    @property
    def execution_lock(self) -> threading.Lock:
        return self._execution_lock

# Global singleton in-memory job store instance
job_store = JobStore()
