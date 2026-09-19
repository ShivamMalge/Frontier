"""Progress reporting that works in-process and inside an RQ worker.

A task calls :func:`report` without knowing which backend is running it:

* under the in-process store, a context-local callback writes to the job record;
* under RQ, the active job's ``meta`` is updated and saved to Redis, where the
  API process reads it back.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

logger = logging.getLogger(__name__)

_reporter: ContextVar[Callable[[float, str], None] | None] = ContextVar(
    "job_progress_reporter", default=None
)


@contextmanager
def reporting_to(callback: Callable[[float, str], None]) -> Iterator[None]:
    """Bind ``callback`` as the progress sink for the current context."""
    token = _reporter.set(callback)
    try:
        yield
    finally:
        _reporter.reset(token)


def report(fraction: float, message: str) -> None:
    """Record progress. Safe to call when nothing is listening."""
    fraction = min(max(float(fraction), 0.0), 1.0)

    callback = _reporter.get()
    if callback is not None:
        callback(fraction, message)
        return

    job = _current_rq_job()
    if job is None:
        return

    job.meta["progress"] = fraction
    job.meta["message"] = message
    try:
        job.save_meta()
    except Exception:
        # Progress is advisory; a broker hiccup must not fail the job itself.
        logger.warning("could not persist progress for job %s", job.id, exc_info=True)


def _current_rq_job():
    try:
        from rq import get_current_job
    except ImportError:
        return None
    try:
        return get_current_job()
    except Exception:
        return None
