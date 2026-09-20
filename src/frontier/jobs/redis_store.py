"""RQ + Redis job store.

What this buys over the in-process store:

* work runs in separate worker processes, so the API stays responsive and can be
  restarted without losing queued or running jobs;
* workers scale horizontally and independently of the web tier -- LSTM training
  is CPU-bound, request handling is not;
* a *running* job can genuinely be stopped, not merely flagged;
* job state survives an API restart, because it lives in Redis rather than in one
  process's memory.
"""

from __future__ import annotations

import logging
from typing import Any

from redis import Redis
from rq import Queue
from rq.command import send_stop_job_command
from rq.exceptions import NoSuchJobError
from rq.job import Job, JobStatus

from frontier.api.schemas.jobs import JobState
from frontier.errors import JobNotFoundError
from frontier.jobs.base import JobRecord, TaskRef

logger = logging.getLogger(__name__)

#: RQ's vocabulary is wider than ours; collapse it onto the API's four states.
_STATE_MAP: dict[str, JobState] = {
    JobStatus.CREATED.value: JobState.QUEUED,
    JobStatus.QUEUED.value: JobState.QUEUED,
    JobStatus.DEFERRED.value: JobState.QUEUED,
    JobStatus.SCHEDULED.value: JobState.QUEUED,
    JobStatus.STARTED.value: JobState.RUNNING,
    JobStatus.FINISHED.value: JobState.SUCCEEDED,
    JobStatus.FAILED.value: JobState.FAILED,
    JobStatus.STOPPED.value: JobState.CANCELLED,
    JobStatus.CANCELED.value: JobState.CANCELLED,
}


class RedisJobStore:
    backend = "redis"

    def __init__(
        self,
        connection: Redis,
        queue_name: str = "pipeline",
        job_timeout: int = 7200,
        retention_seconds: int = 86_400,
    ) -> None:
        self._connection = connection
        self._retention = retention_seconds
        self._queue = Queue(
            queue_name,
            connection=connection,
            default_timeout=job_timeout,
        )

    def submit(self, task: TaskRef) -> JobRecord:
        # Enqueue by dotted path rather than by object: the worker imports the
        # callable itself, so nothing about the function needs to be pickled.
        job = self._queue.enqueue(
            task.path,
            kwargs=task.kwargs,
            result_ttl=self._retention,
            failure_ttl=self._retention,
            meta={"progress": 0.0, "message": "queued"},
        )
        return self._to_record(job)

    def get(self, job_id: str) -> JobRecord:
        return self._to_record(self._fetch(job_id))

    #: States from which there is nothing left to cancel.
    _TERMINAL = frozenset(
        {
            JobStatus.FINISHED.value,
            JobStatus.FAILED.value,
            JobStatus.STOPPED.value,
            JobStatus.CANCELED.value,
        }
    )

    def cancel(self, job_id: str) -> JobRecord:
        job = self._fetch(job_id)
        status = job.get_status(refresh=True)
        raw = status.value if hasattr(status, "value") else str(status)

        if raw in self._TERMINAL:
            # RQ will happily mark a *finished* job as canceled, which hides its
            # result behind a cancelled state. Cancelling something already done
            # is a no-op, matching the in-process store.
            record = self._to_record(job)
            record.message = f"job already {record.state.value}; nothing to cancel"
            return record

        if status == JobStatus.STARTED:
            try:
                # Signals the worker to kill the work horse -- a real stop, which
                # the thread-based store cannot do.
                send_stop_job_command(self._connection, job_id)
            except Exception as exc:  # noqa: BLE001 -- cancellation is best-effort; report what happened
                logger.warning("could not stop running job %s: %s", job_id, exc)
                record = self._to_record(job)
                record.message = f"cancellation requested but the worker did not stop it: {exc}"
                return record
        else:
            try:
                job.cancel()
            except Exception as exc:  # noqa: BLE001 -- see above; a queued job may already be gone
                logger.warning("could not cancel job %s: %s", job_id, exc)

        return self._to_record(self._fetch(job_id))

    def shutdown(self) -> None:
        """Release the client. Queued work deliberately survives -- that is the point."""
        try:
            self._connection.close()
        except Exception:
            logger.debug("redis connection already closed", exc_info=True)

    def _fetch(self, job_id: str) -> Job:
        try:
            return Job.fetch(job_id, connection=self._connection)
        except NoSuchJobError as exc:
            raise JobNotFoundError(f"no job with id '{job_id}'", job_id=job_id) from exc

    def _to_record(self, job: Job) -> JobRecord:
        status = job.get_status(refresh=True)
        raw = status.value if hasattr(status, "value") else str(status)
        state = _STATE_MAP.get(raw, JobState.QUEUED)

        meta: dict[str, Any] = job.meta or {}
        progress = float(meta.get("progress", 0.0))
        message = meta.get("message")

        if state is JobState.SUCCEEDED:
            progress, message = 1.0, message or "complete"
        elif state is JobState.FAILED:
            # exc_info holds the traceback; the last line is the useful summary.
            message = _last_line(job.latest_result()) or message or "failed"

        return JobRecord(
            job_id=job.id,
            state=state,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.ended_at,
            progress=progress,
            message=message,
            result=job.return_value() if state is JobState.SUCCEEDED else None,
        )


def _last_line(result: Any) -> str | None:
    traceback = getattr(result, "exc_string", None)
    if not traceback:
        return None
    lines = [line for line in str(traceback).strip().splitlines() if line.strip()]
    return lines[-1] if lines else None
