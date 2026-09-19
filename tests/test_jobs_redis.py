"""The RQ + Redis job store.

Driven by ``fakeredis`` plus a real ``rq.SimpleWorker``, so the production code
path is genuinely exercised -- enqueue, pick up, execute, persist progress, store
the result, record failures -- without needing a Redis server. ``SimpleWorker``
runs the job in-process instead of forking, which is the one deliberate
divergence; it means these tests cannot cover stopping an already-running job,
noted where that matters.
"""

from __future__ import annotations

import fakeredis
import pytest
from rq import Queue, SimpleWorker

from app.errors import JobNotFoundError
from app.jobs.base import TaskRef
from app.jobs.redis_store import RedisJobStore
from app.schemas.jobs import JobState
from tests.tasks_for_testing import COUNT_UP, ECHO, FAIL

QUEUE = "test-pipeline"


@pytest.fixture
def connection() -> fakeredis.FakeStrictRedis:
    server = fakeredis.FakeServer()
    return fakeredis.FakeStrictRedis(server=server)


@pytest.fixture
def store(connection) -> RedisJobStore:
    return RedisJobStore(connection, queue_name=QUEUE, retention_seconds=3600)


def drain(connection) -> int:
    """Run every queued job to completion, then return. Mirrors `rq worker --burst`."""
    worker = SimpleWorker([Queue(QUEUE, connection=connection)], connection=connection)
    worker.work(burst=True, with_scheduler=False)
    return worker.successful_job_count


class TestEnqueueAndExecute:
    def test_submitted_job_starts_queued(self, store):
        record = store.submit(TaskRef(ECHO, {"value": "hello"}))
        assert record.state is JobState.QUEUED
        assert record.progress == 0.0
        assert store.get(record.job_id).state is JobState.QUEUED

    def test_worker_executes_and_result_is_retrievable(self, store, connection):
        record = store.submit(TaskRef(ECHO, {"value": "hello"}))
        drain(connection)

        done = store.get(record.job_id)
        assert done.state is JobState.SUCCEEDED
        assert done.progress == 1.0
        assert done.result["echoed"] == "hello"
        assert done.started_at is not None
        assert done.finished_at is not None

    def test_task_is_resolved_by_dotted_path_not_pickled_closure(self, store, connection):
        """The whole point of TaskRef: the worker imports the callable itself."""
        job_id = store.submit(TaskRef(ECHO, {"value": "by-path"})).job_id

        from rq.job import Job

        raw = Job.fetch(job_id, connection=connection)
        assert raw.func_name == ECHO

        drain(connection)
        assert store.get(job_id).result["echoed"] == "by-path"

    def test_progress_is_persisted_to_redis(self, store, connection):
        """`report()` must reach Redis via job.meta when running under RQ."""
        record = store.submit(TaskRef(COUNT_UP, {"steps": 5}))
        drain(connection)

        done = store.get(record.job_id)
        assert done.state is JobState.SUCCEEDED
        assert done.result == [0.2, 0.4, 0.6, 0.8, 1.0]

        from rq.job import Job

        meta = Job.fetch(record.job_id, connection=connection).meta
        assert meta["progress"] == 1.0
        assert meta["message"] == "step 5 of 5"


class TestFailures:
    def test_failed_job_reports_the_reason(self, store, connection):
        record = store.submit(TaskRef(FAIL, {"message": "boom in a worker"}))
        drain(connection)

        done = store.get(record.job_id)
        assert done.state is JobState.FAILED
        assert "boom in a worker" in done.message
        assert done.result is None

    def test_unknown_job_id_raises_not_found(self, store):
        with pytest.raises(JobNotFoundError):
            store.get("00000000000000000000000000000000")

    def test_unresolvable_task_path_fails_the_job_not_the_api(self, store, connection):
        """A bad dotted path must surface as a failed job, not a crash on submit."""
        record = store.submit(TaskRef("tests.tasks_for_testing.no_such_function"))
        assert record.state is JobState.QUEUED

        drain(connection)
        assert store.get(record.job_id).state is JobState.FAILED


class TestCancellation:
    def test_queued_job_can_be_cancelled_and_never_runs(self, store, connection):
        record = store.submit(TaskRef(ECHO, {"value": "never"}))

        cancelled = store.cancel(record.job_id)
        assert cancelled.state is JobState.CANCELLED

        # A cancelled job must be skipped rather than executed by the next worker.
        drain(connection)
        assert store.get(record.job_id).state is JobState.CANCELLED
        assert store.get(record.job_id).result is None

    def test_cancelling_an_already_finished_job_is_harmless(self, store, connection):
        record = store.submit(TaskRef(ECHO))
        drain(connection)
        assert store.get(record.job_id).state is JobState.SUCCEEDED

        cancelled = store.cancel(record.job_id)
        # Still finished, and the result is intact. RQ on its own would mark a
        # finished job as canceled and hide the result.
        assert cancelled.state is JobState.SUCCEEDED
        assert "nothing to cancel" in cancelled.message
        assert store.get(record.job_id).result["echoed"] == "ok"

    def test_cancelling_a_failed_job_keeps_it_failed(self, store, connection):
        record = store.submit(TaskRef(FAIL, {"message": "already broken"}))
        drain(connection)

        cancelled = store.cancel(record.job_id)
        assert cancelled.state is JobState.FAILED
        assert store.get(record.job_id).state is JobState.FAILED


class TestDurability:
    def test_queued_work_survives_a_new_store_instance(self, connection):
        """The durability claim: a restarted API process still sees its jobs.

        The in-process store cannot do this -- its records live in the memory of
        the process that submitted them.
        """
        first = RedisJobStore(connection, queue_name=QUEUE)
        job_id = first.submit(TaskRef(ECHO, {"value": "survives"})).job_id
        first.shutdown()

        # A brand-new store, as if the API had been restarted.
        second = RedisJobStore(connection, queue_name=QUEUE)
        assert second.get(job_id).state is JobState.QUEUED

        drain(connection)
        assert second.get(job_id).result["echoed"] == "survives"

    def test_shutdown_does_not_discard_queued_work(self, connection):
        store = RedisJobStore(connection, queue_name=QUEUE)
        job_id = store.submit(TaskRef(ECHO)).job_id
        store.shutdown()

        assert Queue(QUEUE, connection=connection).count == 1
        drain(connection)
        assert RedisJobStore(connection, queue_name=QUEUE).get(job_id).state is (JobState.SUCCEEDED)
