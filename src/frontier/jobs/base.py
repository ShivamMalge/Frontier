"""Job store interface, shared by the in-process and Redis backends.

The central constraint of Phase 2: RQ executes jobs in a **separate process**, so
a job can no longer be a closure. Work is instead named by a :class:`TaskRef` --
an importable dotted path plus JSON-serialisable keyword arguments -- which both
backends resolve the same way. That is what makes the in-process store a faithful
stand-in for the Redis one rather than a divergent code path.
"""

from __future__ import annotations

import datetime as dt
import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from frontier.api.schemas.jobs import JobState


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@dataclass(frozen=True)
class TaskRef:
    """A unit of work: where the callable lives, and what to call it with.

    ``path`` must be importable *in the worker process*, which is a different
    process than the one that enqueued it. Keyword arguments must survive a
    round trip through the broker, so pass plain JSON types -- dicts and lists,
    not Pydantic models or DataFrames.
    """

    path: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def resolve(self) -> Callable[..., Any]:
        module_path, _, attribute = self.path.rpartition(".")
        if not module_path:
            raise ValueError(f"task path must be fully qualified, got {self.path!r}")
        return getattr(importlib.import_module(module_path), attribute)

    def run(self) -> Any:
        return self.resolve()(**self.kwargs)


@dataclass
class JobRecord:
    """Backend-independent view of one job."""

    job_id: str
    state: JobState = JobState.QUEUED
    created_at: dt.datetime = field(default_factory=utcnow)
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    progress: float = 0.0
    message: str | None = None
    result: Any | None = None


class JobStore(Protocol):
    """What the routers depend on. Both backends satisfy this."""

    #: Identifies the backend in /health, so the active one is never a mystery.
    backend: str

    def submit(self, task: TaskRef) -> JobRecord: ...
    def get(self, job_id: str) -> JobRecord: ...
    def cancel(self, job_id: str) -> JobRecord: ...
    def shutdown(self) -> None: ...
