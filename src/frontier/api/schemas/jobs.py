"""Async job contracts.

Phase 1 backs these with an in-process executor. Phase 2 swaps in RQ + Redis
behind the same shapes, so clients written against this contract keep working.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobAccepted(BaseModel):
    """Returned with HTTP 202 when work is handed to the queue."""

    job_id: str
    state: JobState
    status_url: str
    result_url: str


class JobStatus(BaseModel):
    job_id: str
    state: JobState
    created_at: dt.datetime
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str | None = Field(default=None, description="Current step, or the failure reason.")
    result: Any | None = Field(default=None, description="Populated once the state is 'succeeded'.")
