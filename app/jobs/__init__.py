"""Job execution.

Two interchangeable backends behind one protocol:

* ``memory`` -- a thread pool in the API process. No dependencies, no durability.
* ``redis``  -- RQ workers in separate processes. Durable, scalable, interruptible.

Which one is active is decided by ``SO_JOB_BACKEND``:

* ``auto`` (default) tries Redis and falls back to memory if it is unreachable,
  which keeps ``pytest`` and ``python main.py`` working on a bare checkout;
* ``redis`` requires Redis and fails at startup if it is missing -- use this in
  production, where a silent fallback to a single-process queue would be worse
  than not booting;
* ``memory`` forces the in-process store.

``GET /health`` reports the backend actually in use, so a fallback is never silent.
"""

from __future__ import annotations

import logging
import threading

from app.jobs.base import JobRecord, JobStore, TaskRef
from app.jobs.memory import InMemoryJobStore
from app.jobs.progress import report, reporting_to
from app.settings import get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "InMemoryJobStore",
    "JobRecord",
    "JobStore",
    "TaskRef",
    "create_job_store",
    "get_job_store",
    "report",
    "reporting_to",
    "set_job_store",
]

_store: JobStore | None = None
_lock = threading.Lock()


def create_job_store() -> JobStore:
    """Build the store named by configuration, honouring the ``auto`` fallback."""
    settings = get_settings()
    requested = settings.job_backend

    if requested == "memory":
        return _memory_store()

    try:
        return _redis_store()
    except Exception as exc:
        if requested == "redis":
            raise RuntimeError(
                f"SO_JOB_BACKEND=redis but Redis at {settings.redis_url} is unreachable: {exc}"
            ) from exc
        logger.warning(
            "Redis at %s is unreachable (%s); falling back to the in-process job store. "
            "Jobs will not survive a restart and cannot be shared across workers.",
            settings.redis_url,
            exc,
        )
        return _memory_store()


def _memory_store() -> JobStore:
    settings = get_settings()
    return InMemoryJobStore(
        workers=settings.job_workers,
        retention_seconds=settings.job_retention_seconds,
    )


def _redis_store() -> JobStore:
    from redis import Redis

    from app.jobs.redis_store import RedisJobStore

    settings = get_settings()
    connection = Redis.from_url(
        settings.redis_url,
        socket_connect_timeout=settings.redis_connect_timeout_seconds,
        socket_timeout=settings.redis_socket_timeout_seconds,
    )
    connection.ping()  # fail here rather than on the first enqueue
    logger.info("job backend: redis at %s (queue %r)", settings.redis_url, settings.queue_name)
    return RedisJobStore(
        connection,
        queue_name=settings.queue_name,
        job_timeout=settings.job_timeout_seconds,
        retention_seconds=settings.job_retention_seconds,
    )


def get_job_store() -> JobStore:
    global _store
    with _lock:
        if _store is None:
            _store = create_job_store()
        return _store


def set_job_store(store: JobStore | None) -> None:
    """Replace the active store. Used by the tests and by the app lifespan."""
    global _store
    with _lock:
        _store = store
