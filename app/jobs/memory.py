"""In-process job store.

Kept as the default for development, tests and the CLI, where standing up Redis
would be friction for no benefit. Its limits are real and are why Phase 2 exists:
records live in one process's memory, vanish on restart, are invisible to other
workers, and a running job cannot truly be interrupted.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

from app.errors import JobNotFoundError
from app.jobs.base import JobRecord, TaskRef, utcnow
from app.jobs.progress import reporting_to
from app.schemas.jobs import JobState

logger = logging.getLogger(__name__)


class InMemoryJobStore:
    backend = "memory"

    def __init__(self, workers: int, retention_seconds: int) -> None:
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")
        self._retention = dt.timedelta(seconds=retention_seconds)
        self._lock = threading.Lock()
        self._records: dict[str, JobRecord] = {}
        self._futures: dict[str, Future[None]] = {}
        self._cancelled: set[str] = set()

    def submit(self, task: TaskRef) -> JobRecord:
        self._evict_expired()
        job_id = uuid.uuid4().hex
        record = JobRecord(job_id=job_id, message="queued")

        with self._lock:
            self._records[job_id] = record

        self._futures[job_id] = self._executor.submit(self._run, job_id, task)
        return record

    def _run(self, job_id: str, task: TaskRef) -> None:
        with self._lock:
            record = self._records[job_id]
            if job_id in self._cancelled:
                record.state = JobState.CANCELLED
                record.finished_at = utcnow()
                record.message = "cancelled before it started"
                return
            record.state = JobState.RUNNING
            record.started_at = utcnow()
            record.message = "starting"

        def on_progress(fraction: float, message: str) -> None:
            with self._lock:
                current = self._records[job_id]
                current.progress = fraction
                current.message = message

        try:
            with reporting_to(on_progress):
                result = task.run()
        except Exception as exc:
            logger.exception("job %s failed", job_id)
            with self._lock:
                record = self._records[job_id]
                record.state = JobState.FAILED
                record.finished_at = utcnow()
                record.message = f"{type(exc).__name__}: {exc}"
            return

        with self._lock:
            record = self._records[job_id]
            record.state = JobState.SUCCEEDED
            record.finished_at = utcnow()
            record.progress = 1.0
            record.message = "complete"
            record.result = result

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._records.get(job_id)
        if record is None:
            raise JobNotFoundError(f"no job with id '{job_id}'", job_id=job_id)
        return record

    def cancel(self, job_id: str) -> JobRecord:
        record = self.get(job_id)
        with self._lock:
            self._cancelled.add(job_id)

        future = self._futures.get(job_id)
        if future is not None and future.cancel():
            with self._lock:
                record.state = JobState.CANCELLED
                record.finished_at = utcnow()
                record.message = "cancelled"
        elif record.state is JobState.RUNNING:
            # Threads cannot be interrupted. The Redis backend can genuinely stop
            # a running job because its work happens in a separate process.
            record.message = "cancellation requested; the job is already running"
        return record

    def _evict_expired(self) -> None:
        cutoff = utcnow() - self._retention
        with self._lock:
            stale = [
                job_id
                for job_id, record in self._records.items()
                if record.finished_at is not None and record.finished_at < cutoff
            ]
            for job_id in stale:
                self._records.pop(job_id, None)
                self._futures.pop(job_id, None)
                self._cancelled.discard(job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
