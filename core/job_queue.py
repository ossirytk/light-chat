"""Simple in-memory job store for long-running RAG web operations.

Jobs run in background threads. Route handlers poll for status via HTMX
(`hx-trigger="every 2s"`). The job status endpoint stops including the
polling trigger once the job reaches a terminal state (done or error).
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class Job:
    """In-memory representation of a background job."""

    __slots__ = ("error", "finished_at", "id", "result", "started_at", "status")

    def __init__(self, job_id: str) -> None:
        self.id = job_id
        self.status: str = "pending"
        self.result: Any = None
        self.error: str | None = None
        self.started_at: float = time.monotonic()
        self.finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        elapsed = round((self.finished_at or time.monotonic()) - self.started_at, 2)
        return {
            "id": self.id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "elapsed_s": elapsed,
        }


class JobStore:
    """Thread-safe store for background jobs."""

    MAX_JOBS: int = 50

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable[..., Any], *args: object, **kwargs: object) -> str:
        """Submit a callable as a background job; returns a job_id immediately."""
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id)
        with self._lock:
            self._jobs[job_id] = job
            self._evict_old()

        def _run() -> None:
            job.status = "running"
            try:
                job.result = fn(*args, **kwargs)
                job.status = "done"
            except Exception as exc:
                job.error = str(exc)
                job.status = "error"
            finally:
                job.finished_at = time.monotonic()

        threading.Thread(target=_run, daemon=True).start()
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Return job state dict, or None if job_id is unknown."""
        with self._lock:
            job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    def _evict_old(self) -> None:
        """Remove oldest finished jobs when over the cap (called under lock)."""
        if len(self._jobs) <= self.MAX_JOBS:
            return
        finished = [j for j in self._jobs.values() if j.status in {"done", "error"}]
        finished.sort(key=lambda j: j.finished_at or 0)
        for j in finished[: len(self._jobs) - self.MAX_JOBS]:
            del self._jobs[j.id]
