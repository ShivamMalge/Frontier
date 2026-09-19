"""Multi-process execution against a real Redis server.

Every other Redis test uses ``fakeredis`` with ``SimpleWorker``, which runs the
job in the calling process. That covers the RQ protocol but not the thing Phase 2
actually exists for: work leaving the API process entirely.

This module starts a real Redis (via ``redislite``, no root or Docker needed) and
launches ``python -m app.worker`` as a genuine subprocess, so it also exercises the
worker entry point itself. It is skipped when ``redislite`` is unavailable.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

import pytest
import redis

from app.jobs.base import TaskRef
from app.jobs.redis_store import RedisJobStore
from app.schemas.jobs import JobState
from tests.tasks_for_testing import COUNT_UP, ECHO, FAIL

redislite = pytest.importorskip("redislite", reason="needs redislite for a real Redis server")

QUEUE = "test-subprocess"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def redis_url():
    """A real Redis server on a private unix socket, torn down afterwards."""
    workdir = tempfile.mkdtemp(prefix="so-test-redis-")
    server = redislite.Redis(os.path.join(workdir, "redis.db"))
    try:
        yield f"unix://{server.socket_file}?db=0"
    finally:
        server.shutdown()


@pytest.fixture
def store(redis_url) -> RedisJobStore:
    connection = redis.Redis.from_url(redis_url)
    return RedisJobStore(connection, queue_name=QUEUE, retention_seconds=3600)


def run_worker(redis_url: str, timeout: float = 120, **env: str) -> subprocess.CompletedProcess:
    """Run the real worker entry point in burst mode and wait for it to drain."""
    return subprocess.run(
        [sys.executable, "-m", "app.worker", "--url", redis_url, "--queues", QUEUE, "--burst"],
        capture_output=True,
        text=True,
        # The exit code is what the caller asserts on; do not raise on it here.
        check=False,
        timeout=timeout,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": REPO_ROOT, **env},
    )


def await_terminal(store: RedisJobStore, job_id: str, timeout: float = 60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = store.get(job_id)
        if record.state not in {JobState.QUEUED, JobState.RUNNING}:
            return record
        time.sleep(0.05)
    pytest.fail(f"job {job_id} never reached a terminal state")


def test_job_executes_in_a_different_process(store, redis_url):
    """The result must be produced by a pid other than this one."""
    record = store.submit(TaskRef(ECHO, {"value": "cross-process"}))
    assert store.get(record.job_id).state is JobState.QUEUED

    completed = run_worker(redis_url)
    assert completed.returncode == 0, completed.stdout + completed.stderr

    done = await_terminal(store, record.job_id)
    assert done.state is JobState.SUCCEEDED
    assert done.result["echoed"] == "cross-process"
    assert done.result["pid"] != str(os.getpid()), "job ran in the API process"


def test_progress_written_by_another_process_is_readable(store, redis_url):
    record = store.submit(TaskRef(COUNT_UP, {"steps": 4}))
    run_worker(redis_url)

    done = await_terminal(store, record.job_id)
    assert done.state is JobState.SUCCEEDED
    assert done.progress == 1.0
    assert done.result == [0.25, 0.5, 0.75, 1.0]


def test_worker_reports_task_failure_back_through_redis(store, redis_url):
    record = store.submit(TaskRef(FAIL, {"message": "failed in a subprocess"}))
    completed = run_worker(redis_url)
    # A failing job is not a failing worker; the worker drains and exits cleanly.
    assert completed.returncode == 0

    done = await_terminal(store, record.job_id)
    assert done.state is JobState.FAILED
    assert "failed in a subprocess" in done.message


def test_worker_exits_nonzero_when_redis_is_unreachable():
    completed = subprocess.run(
        [sys.executable, "-m", "app.worker", "--url", "redis://127.0.0.1:1/0", "--burst"],
        capture_output=True,
        text=True,
        # A non-zero exit is exactly what this test is checking for.
        check=False,
        timeout=60,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
    )
    assert completed.returncode == 1
    assert "cannot reach Redis" in completed.stdout + completed.stderr


def test_full_pipeline_crosses_the_process_boundary(store, redis_url):
    """The real task, with the real request serialisation, in a real worker.

    The worker is told to use the synthetic data source through its environment,
    because a separate process cannot see this suite's fixtures -- and a test must
    not depend on the network.
    """
    from app.schemas.pipeline import PipelineRequest
    from app.tasks import RUN_PIPELINE

    request = PipelineRequest(
        tickers=["AAA", "BBB", "CCC"],
        backend="naive",
        start="2018-01-01",
        end="2021-08-01",
    )
    record = store.submit(TaskRef(RUN_PIPELINE, {"request": request.model_dump(mode="json")}))

    completed = run_worker(redis_url, timeout=300, SO_MARKET_DATA_SOURCE="synthetic")
    assert completed.returncode == 0, completed.stdout + completed.stderr

    done = await_terminal(store, record.job_id, timeout=120)
    assert done.state is JobState.SUCCEEDED, done.message

    result = done.result
    assert result["tickers"] == ["AAA", "BBB", "CCC"]
    from app.services.optimization import ALL_STRATEGIES

    assert len(result["performance"]) == len(ALL_STRATEGIES)
    assert result["selected_strategy"] in result["weights"]["columns"]
