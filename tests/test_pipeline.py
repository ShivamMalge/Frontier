"""Pipeline runs and the job lifecycle."""

from __future__ import annotations

import time

import pytest

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def _await_terminal(client, job_id: str, timeout: float = 60.0) -> dict:
    """Poll until the job leaves the queued/running states."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/pipeline/runs/{job_id}/result").json()
        if body["state"] not in {"queued", "running"}:
            return body
        time.sleep(0.02)
    pytest.fail(f"job {job_id} did not finish within {timeout}s")


def test_submit_returns_202_with_pollable_urls(client, window):
    response = client.post(
        "/api/v1/pipeline/runs", json={"tickers": TICKERS, "backend": "naive", **window}
    )
    assert response.status_code == 202

    body = response.json()
    assert body["state"] in {"queued", "running"}
    assert body["status_url"] == f"/api/v1/pipeline/runs/{body['job_id']}"
    assert response.headers["Location"] == body["status_url"]


def test_status_endpoint_omits_the_result_payload(client, window):
    job_id = client.post(
        "/api/v1/pipeline/runs", json={"tickers": TICKERS, "backend": "naive", **window}
    ).json()["job_id"]
    _await_terminal(client, job_id)

    status = client.get(f"/api/v1/pipeline/runs/{job_id}").json()
    assert status["state"] == "succeeded"
    assert status["progress"] == 1.0
    # Polling must stay cheap; the full result lives behind /result.
    assert "result" not in status


def test_completed_run_produces_a_coherent_result(client, window):
    job_id = client.post(
        "/api/v1/pipeline/runs",
        json={"tickers": TICKERS, "backend": "naive", "risk_tolerance": 0.5, **window},
    ).json()["job_id"]

    body = _await_terminal(client, job_id)
    assert body["state"] == "succeeded", body["message"]

    result = body["result"]
    assert result["tickers"] == TICKERS
    assert result["backend"] == "naive"
    assert len(result["forecast_metrics"]) == len(TICKERS)
    assert len(result["performance"]) == 6

    assert result["selected_strategy"] in result["weights"]["columns"]
    assert sum(result["selected_weights"].values()) == pytest.approx(1.0, abs=1e-9)

    for row in result["performance"]:
        assert -1.0 < row["annual_return"] < 2.0, row
        assert 0.0 < row["annual_volatility"] < 1.0, row


def test_risk_tolerance_picks_along_the_volatility_ranking(client, window):
    def run(tolerance: float) -> dict:
        job_id = client.post(
            "/api/v1/pipeline/runs",
            json={
                "tickers": TICKERS,
                "backend": "naive",
                "risk_tolerance": tolerance,
                **window,
            },
        ).json()["job_id"]
        return _await_terminal(client, job_id)["result"]

    low, high = run(0.0), run(1.0)
    vols = {row["strategy"]: row["annual_volatility"] for row in low["performance"]}

    assert vols[low["selected_strategy"]] == min(vols.values())
    assert vols[high["selected_strategy"]] == max(vols.values())


def test_unknown_job_is_a_404(client):
    response = client.get("/api/v1/pipeline/runs/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "job_not_found"


def test_failing_job_reports_the_reason_without_crashing_the_api(client, window):
    """A job body that raises must surface as state=failed, not a 500."""
    from app.services import pipeline as pipeline_service

    def boom(request, report=None):
        raise RuntimeError("synthetic pipeline failure")

    original = pipeline_service.run
    pipeline_service.run = boom
    try:
        job_id = client.post(
            "/api/v1/pipeline/runs", json={"tickers": TICKERS, "backend": "naive", **window}
        ).json()["job_id"]
        body = _await_terminal(client, job_id)
    finally:
        pipeline_service.run = original

    assert body["state"] == "failed"
    assert "synthetic pipeline failure" in body["message"]
    assert client.get("/health").status_code == 200


def test_cancelling_a_queued_job_marks_it_cancelled(client):
    """A job still waiting for a worker can be cancelled outright."""
    from app import jobs
    from app.jobs import TaskRef
    from app.settings import get_settings
    from tests.tasks_for_testing import BLOCK_UNTIL_RELEASED, ECHO, signal_path

    store = jobs.get_job_store()
    token = "cancel-queued"
    release = signal_path(token)
    release.unlink(missing_ok=True)
    started = signal_path(f"{token}-started")
    started.unlink(missing_ok=True)

    # Saturate every worker so the next submission is guaranteed to sit queued.
    workers = get_settings().job_workers
    for _ in range(workers):
        store.submit(TaskRef(BLOCK_UNTIL_RELEASED, {"token": token}))

    deadline = time.monotonic() + 10
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started.exists(), "no worker picked up the blocking task"

    queued = store.submit(TaskRef(ECHO, {"value": "must never run"}))
    try:
        body = client.delete(f"/api/v1/pipeline/runs/{queued.job_id}").json()
        assert body["state"] == "cancelled"
        assert store.get(queued.job_id).result is None
    finally:
        release.touch()


def test_job_records_expire_after_their_retention_window(client):
    """Finished records are evicted so a long-lived process does not leak them."""
    import datetime as dt

    from app.errors import JobNotFoundError
    from app.jobs import TaskRef
    from app.jobs.memory import InMemoryJobStore
    from tests.tasks_for_testing import ECHO

    store = InMemoryJobStore(workers=1, retention_seconds=0)
    try:
        record = store.submit(TaskRef(ECHO))
        deadline = time.monotonic() + 10
        while store.get(record.job_id).finished_at is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert store.get(record.job_id).state == "succeeded"

        # Age the record past retention, then trigger eviction with a new submit.
        store.get(record.job_id).finished_at = dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)
        store.submit(TaskRef(ECHO))

        with pytest.raises(JobNotFoundError):
            store.get(record.job_id)
    finally:
        store.shutdown()
