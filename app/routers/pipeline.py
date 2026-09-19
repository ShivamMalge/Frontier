"""Pipeline runs, exposed as pollable jobs.

The full pipeline can take tens of minutes with ``keras_lstm``, so it is never
run inside a request. Submitting returns 202 with a job id; the client polls for
progress and collects the result when the state reaches ``succeeded``.

Phase 2 replaces the in-process executor behind :func:`app.jobs.get_job_store`
with RQ + Redis. These three endpoints stay as they are.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.jobs import TaskRef, get_job_store
from app.schemas.job_results import JobResult
from app.schemas.jobs import JobAccepted, JobStatus
from app.schemas.pipeline import PipelineRequest
from app.settings import get_settings
from app.tasks import RUN_PIPELINE

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _status_payload(record) -> JobStatus:
    return JobStatus(
        job_id=record.job_id,
        state=record.state,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        progress=record.progress,
        message=record.message,
        result=record.result,
    )


@router.post(
    "/runs",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a full pipeline run",
)
def submit(request: PipelineRequest, response: Response) -> JobAccepted:
    prefix = get_settings().api_v1_prefix
    # A dotted path plus JSON kwargs, not a closure: with the Redis backend the
    # work is executed by a different process, which imports the task itself.
    record = get_job_store().submit(
        TaskRef(RUN_PIPELINE, {"request": request.model_dump(mode="json")})
    )

    status_url = f"{prefix}/pipeline/runs/{record.job_id}"
    response.headers["Location"] = status_url
    return JobAccepted(
        job_id=record.job_id,
        state=record.state,
        status_url=status_url,
        result_url=f"{status_url}/result",
    )


@router.get(
    "/runs/{job_id}",
    response_model=JobStatus,
    response_model_exclude={"result"},
    summary="Poll a run's progress",
)
def get_status(job_id: str) -> JobStatus:
    return _status_payload(get_job_store().get(job_id))


@router.get(
    "/runs/{job_id}/result",
    # JobResult, not JobStatus: it narrows `result` to the three shapes tasks
    # actually return, so clients get a published schema instead of `any`.
    response_model=JobResult,
    summary="Fetch a completed run, including its result",
)
def get_result(job_id: str) -> JobStatus:
    return _status_payload(get_job_store().get(job_id))


@router.delete("/runs/{job_id}", response_model=JobStatus, summary="Request cancellation")
def cancel(job_id: str) -> JobStatus:
    return _status_payload(get_job_store().cancel(job_id))
