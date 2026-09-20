"""The API driven end-to-end against the RQ + Redis job store.

These are the tests that justify Phase 2: a pipeline request submitted over HTTP
must survive serialisation to the broker, execute in a worker that never saw the
request object, and come back through the same HTTP contract as before.
"""

from __future__ import annotations

import fakeredis
import pytest
from fastapi.testclient import TestClient
from rq import Queue, SimpleWorker

from frontier import jobs
from frontier.api.main import create_app
from frontier.jobs.redis_store import RedisJobStore

QUEUE = "test-api-pipeline"
TICKERS = ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture
def connection() -> fakeredis.FakeStrictRedis:
    return fakeredis.FakeStrictRedis(server=fakeredis.FakeServer())


@pytest.fixture
def redis_client(connection) -> TestClient:
    """A TestClient whose job store is RQ-backed."""
    jobs.set_job_store(RedisJobStore(connection, queue_name=QUEUE, retention_seconds=3600))
    with TestClient(create_app()) as client:
        yield client
    jobs.set_job_store(None)


def drain(connection) -> None:
    worker = SimpleWorker([Queue(QUEUE, connection=connection)], connection=connection)
    worker.work(burst=True, with_scheduler=False)


def test_health_reports_the_redis_backend(redis_client):
    """A fallback to the in-process store must never be silent."""
    assert redis_client.get("/health").json()["job_backend"] == "redis"


def test_submitted_run_is_queued_then_executed_by_a_worker(redis_client, connection, window):
    accepted = redis_client.post(
        "/api/v1/pipeline/runs",
        json={"tickers": TICKERS, "backend": "naive", **window},
    )
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]

    # Nothing has run yet: the API process does no work of its own.
    assert redis_client.get(f"/api/v1/pipeline/runs/{job_id}").json()["state"] == "queued"
    assert Queue(QUEUE, connection=connection).count == 1

    drain(connection)

    body = redis_client.get(f"/api/v1/pipeline/runs/{job_id}/result").json()
    assert body["state"] == "succeeded", body["message"]

    result = body["result"]
    assert result["tickers"] == TICKERS
    from frontier.services.optimization import ALL_STRATEGIES

    assert len(result["performance"]) == len(ALL_STRATEGIES)
    assert result["selected_strategy"] in result["weights"]["columns"]
    assert sum(result["selected_weights"].values()) == pytest.approx(1.0, abs=1e-9)


def test_request_survives_the_serialisation_round_trip(redis_client, connection, window):
    """Non-default parameters must reach the worker intact.

    The worker receives only JSON, so anything the request carries has to survive
    ``model_dump`` on one side and ``model_validate`` on the other.
    """
    job_id = redis_client.post(
        "/api/v1/pipeline/runs",
        json={
            "tickers": ["AAA", "BBB"],
            "backend": "naive",
            "train_split": 0.8,
            "lookback_window": 30,
            "risk_free_rate": 0.04,
            "strategies": ["HRP", "RiskParity"],
            **window,
        },
    ).json()["job_id"]

    drain(connection)
    result = redis_client.get(f"/api/v1/pipeline/runs/{job_id}/result").json()["result"]

    assert result["tickers"] == ["AAA", "BBB"]
    assert result["weights"]["columns"] == ["HRP", "RiskParity"]
    assert {row["strategy"] for row in result["performance"]} == {"HRP", "RiskParity"}


def test_progress_written_by_the_worker_is_visible_to_the_api(redis_client, connection, window):
    job_id = redis_client.post(
        "/api/v1/pipeline/runs",
        json={"tickers": TICKERS, "backend": "naive", **window},
    ).json()["job_id"]

    drain(connection)

    status = redis_client.get(f"/api/v1/pipeline/runs/{job_id}").json()
    assert status["progress"] == 1.0
    assert status["message"] == "complete"
    # Polling stays cheap: the result is only on the /result route.
    assert "result" not in status


def test_failing_run_surfaces_as_failed_not_a_500(redis_client, connection, window):
    job_id = redis_client.post(
        "/api/v1/pipeline/runs",
        json={"tickers": ["NOPE"], "backend": "naive", **window},
    ).json()["job_id"]

    drain(connection)

    body = redis_client.get(f"/api/v1/pipeline/runs/{job_id}/result").json()
    assert body["state"] == "failed"
    assert body["message"]
    assert redis_client.get("/health").status_code == 200


def test_queued_run_can_be_cancelled_before_a_worker_takes_it(redis_client, connection, window):
    job_id = redis_client.post(
        "/api/v1/pipeline/runs",
        json={"tickers": TICKERS, "backend": "naive", **window},
    ).json()["job_id"]

    assert redis_client.delete(f"/api/v1/pipeline/runs/{job_id}").json()["state"] == "cancelled"

    drain(connection)
    assert redis_client.get(f"/api/v1/pipeline/runs/{job_id}").json()["state"] == "cancelled"


def test_jobs_outlive_the_api_process(redis_client, connection, window):
    """Restarting the API must not lose submitted work.

    This is the capability the in-process store cannot provide, and the main
    reason for the move to RQ.
    """
    job_id = redis_client.post(
        "/api/v1/pipeline/runs",
        json={"tickers": TICKERS, "backend": "naive", **window},
    ).json()["job_id"]

    # Simulate a restart: tear down the app and build a fresh one on the same Redis.
    jobs.set_job_store(RedisJobStore(connection, queue_name=QUEUE, retention_seconds=3600))
    with TestClient(create_app()) as restarted:
        assert restarted.get(f"/api/v1/pipeline/runs/{job_id}").json()["state"] == "queued"
        drain(connection)
        body = restarted.get(f"/api/v1/pipeline/runs/{job_id}/result").json()
        assert body["state"] == "succeeded"
        assert body["result"]["tickers"] == TICKERS
