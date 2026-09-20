"""Experiment tracking.

Two properties matter more than the logging itself.

First, **tracking must never break the work**. A run is telemetry; failing a
forty-minute backtest because a metric would not serialise is a bad trade. Several
tests below deliberately break the tracking store and assert the pipeline still
completes.

Second, a logged run must be **enough to reproduce or dispute the result**: the
parameters, the seed, the data vintage, the code commit, and the baseline comparison
alongside the legacy metric. That is the gap this phase closes -- the original
project's "93%+ accuracy" had none of it.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from frontier import jobs
from frontier.api.main import create_app
from frontier.services import tracking
from frontier.settings import get_settings

TICKERS = ["AAA", "BBB", "CCC"]
WINDOW = {"start": "2018-01-01", "end": "2021-08-01"}


@pytest.fixture
def tracked(tmp_path, monkeypatch):
    """A client with MLflow enabled against a throwaway SQLite store."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("SO_MLFLOW_ENABLED", "true")
    monkeypatch.setenv("SO_MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("SO_MLFLOW_EXPERIMENT", "test-frontier")
    get_settings.cache_clear()

    jobs.set_job_store(None)
    with TestClient(create_app()) as client:
        yield client, uri
    jobs.set_job_store(None)
    get_settings.cache_clear()


def await_result(client, job_id: str, timeout: float = 300.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/pipeline/runs/{job_id}/result").json()
        if body["state"] not in {"queued", "running"}:
            return body
        time.sleep(0.05)
    pytest.fail("job did not finish")


def run_pipeline(client, **overrides) -> dict:
    payload = {"tickers": TICKERS, "backend": "naive", **WINDOW, **overrides}
    response = client.post("/api/v1/pipeline/runs", json=payload)
    assert response.status_code == 202
    body = await_result(client, response.json()["job_id"])
    assert body["state"] == "succeeded", body["message"]
    return body["result"]


def fetch(uri: str, run_id: str):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=uri).get_run(run_id)


class TestDisabledByDefault:
    def test_nothing_is_tracked_unless_enabled(self, client):
        result = run_pipeline(client)
        assert result["tracking_run_id"] is None

    def test_status_reports_disabled_with_a_hint(self, client):
        body = client.get("/api/v1/tracking/status").json()
        assert body["enabled"] is False
        assert "SO_MLFLOW_ENABLED" in body["detail"]

    def test_runs_listing_is_empty_rather_than_an_error(self, client):
        body = client.get("/api/v1/tracking/runs").json()
        assert body == {"runs": [], "total": 0}

    def test_the_context_manager_is_inert(self, client):
        with tracking.track("noop") as run:
            run.log_params({"a": 1})
            run.log_metrics({"b": 2})
            run.log_table([{"c": 3}], "t.json")
            assert not run.active
            assert run.logged_metrics == {}


class TestPipelineRun:
    def test_a_run_id_is_returned_and_resolvable(self, tracked):
        client, uri = tracked
        result = run_pipeline(client)

        assert result["tracking_run_id"]
        run = fetch(uri, result["tracking_run_id"])
        assert run.info.status == "FINISHED"

    def test_parameters_needed_to_reproduce_are_logged(self, tracked):
        client, uri = tracked
        result = run_pipeline(client, seed=7, lags=5)
        params = fetch(uri, result["tracking_run_id"]).data.params

        assert params["backend"] == "naive"
        assert params["seed"] == "7"
        assert params["lags"] == "5"
        assert params["target"] == "return"
        assert params["n_tickers"] == str(len(TICKERS))
        assert params["start"] == WINDOW["start"]
        assert params["end"] == WINDOW["end"]

    def test_the_baseline_comparison_is_logged_beside_the_legacy_metric(self, tracked):
        """The specific gap this phase closes.

        "93% accuracy" was unfalsifiable because nothing recorded what it measured.
        Both numbers are now on the same run, so the discrepancy is visible.
        """
        client, uri = tracked
        result = run_pipeline(client)
        metrics = fetch(uri, result["tracking_run_id"]).data.metrics

        assert "mase_vs_naive_mean" in metrics
        assert "legacy_approximate_accuracy_mean" in metrics
        # The naive backend: MASE exactly 1.0, legacy metric flatteringly high.
        assert metrics["mase_vs_naive_mean"] == pytest.approx(1.0, abs=1e-9)
        assert metrics["legacy_approximate_accuracy_mean"] > 90.0

    def test_per_strategy_metrics_are_logged_under_their_own_prefix(self, tracked):
        client, uri = tracked
        result = run_pipeline(client)
        metrics = fetch(uri, result["tracking_run_id"]).data.metrics

        assert "HRP__sharpe" in metrics
        assert "Markowitz_MinVar__annual_volatility" in metrics
        assert metrics["strategies"] == len(result["weights"]["columns"])

    def test_per_ticker_detail_is_logged_as_an_artifact(self, tracked):
        """Ten strategies times five metrics is a table, not fifty scalars."""
        from mlflow.tracking import MlflowClient

        client, uri = tracked
        result = run_pipeline(client)

        artifacts = {
            a.path for a in MlflowClient(tracking_uri=uri).list_artifacts(result["tracking_run_id"])
        }
        assert "forecast_metrics.json" in artifacts
        assert "strategy_performance.json" in artifacts

    def test_the_code_commit_is_tagged(self, tracked):
        """Without it a run records what was measured but not what measured it."""
        client, uri = tracked
        result = run_pipeline(client)
        tags = fetch(uri, result["tracking_run_id"]).data.tags

        if tracking.git_commit():
            assert tags["git_commit"] == tracking.git_commit()
        assert tags["kind"] == "pipeline"
        # The *configured* source, which is what a reader needs to know. This suite
        # stubs the download function rather than switching the setting, so the tag
        # reports yfinance even though no network call happens.
        assert tags["market_data_source"] == get_settings().market_data_source

    def test_a_null_metric_is_skipped_not_logged_as_zero(self, tracked):
        """Directional accuracy is null for a flat forecast; zero would read as
        "always wrong" rather than "never guessed"."""
        client, uri = tracked
        result = run_pipeline(client)
        metrics = fetch(uri, result["tracking_run_id"]).data.metrics
        assert "directional_accuracy_mean" not in metrics


class TestBacktestRun:
    def test_backtest_is_tracked_separately(self, tracked):
        client, uri = tracked
        submitted = client.post(
            "/api/v1/backtest/runs",
            json={
                "tickers": TICKERS,
                **WINDOW,
                "lookback": 120,
                "rebalance_every": 60,
                "strategies": ["Markowitz_MinVar", "HRP"],
            },
        )
        body = await_result(client, submitted.json()["job_id"])
        assert body["state"] == "succeeded", body["message"]

        run_id = body["result"]["tracking_run_id"]
        assert run_id
        run = fetch(uri, run_id)

        assert run.data.tags["kind"] == "backtest"
        assert run.data.params["lookback"] == "120"
        assert run.data.params["cost_bps"] == "10.0"
        assert "HRP__Sharpe" in run.data.metrics
        assert "HRP__Cost_Drag" in run.data.metrics
        assert "sharpe_mean" in run.data.metrics


class TestRunListing:
    def test_runs_are_listed_newest_first(self, tracked):
        client, _ = tracked
        run_pipeline(client)
        run_pipeline(client, seed=2)

        body = client.get("/api/v1/tracking/runs").json()
        assert body["total"] == 2
        assert all(run["kind"] == "pipeline" for run in body["runs"])
        assert body["runs"][0]["started_at"] >= body["runs"][1]["started_at"]

    def test_listing_can_be_filtered_by_kind(self, tracked):
        client, _ = tracked
        run_pipeline(client)
        assert client.get("/api/v1/tracking/runs?kind=pipeline").json()["total"] == 1
        assert client.get("/api/v1/tracking/runs?kind=backtest").json()["total"] == 0

    def test_status_counts_runs(self, tracked):
        client, _ = tracked
        run_pipeline(client)
        body = client.get("/api/v1/tracking/status").json()
        assert body["enabled"] is True
        assert body["run_count"] == 1
        assert body["ui_command"].startswith("mlflow ui")


class TestTrackingNeverBreaksTheWork:
    """A pipeline must not fail because telemetry did."""

    def test_an_unreachable_store_does_not_fail_the_run(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SO_MLFLOW_ENABLED", "true")
        # A path that cannot be created.
        monkeypatch.setenv("SO_MLFLOW_TRACKING_URI", "sqlite:////proc/nope/mlflow.db")
        get_settings.cache_clear()
        jobs.set_job_store(None)
        try:
            with TestClient(create_app()) as client:
                result = run_pipeline(client)
            assert result["tracking_run_id"] is None
            assert len(result["performance"]) > 0
        finally:
            jobs.set_job_store(None)
            get_settings.cache_clear()

    def test_a_failing_log_call_does_not_fail_the_run(self, tracked, monkeypatch):
        client, _ = tracked

        def explode(*args, **kwargs):
            raise RuntimeError("synthetic tracking failure")

        import mlflow

        monkeypatch.setattr(mlflow, "log_metrics", explode)
        monkeypatch.setattr(mlflow, "log_table", explode)

        result = run_pipeline(client)
        assert len(result["performance"]) > 0
        assert result["tracking_run_id"]

    def test_a_failing_run_is_marked_failed_not_left_open(self, tracked, monkeypatch):
        client, uri = tracked
        from frontier.services import pipeline as pipeline_service

        captured: dict[str, str] = {}
        original = pipeline_service._run

        def failing(request, report, run_handle):
            captured["run_id"] = run_handle.run_id or ""
            raise RuntimeError("synthetic pipeline failure")

        monkeypatch.setattr(pipeline_service, "_run", failing)

        response = client.post(
            "/api/v1/pipeline/runs",
            json={"tickers": TICKERS, "backend": "naive", **WINDOW},
        )
        body = await_result(client, response.json()["job_id"])
        assert body["state"] == "failed"

        monkeypatch.setattr(pipeline_service, "_run", original)
        assert fetch(uri, captured["run_id"]).info.status == "FAILED"


class TestHelpers:
    def test_aggregate_skips_missing_values(self):
        assert tracking.aggregate([1.0, None, 3.0], "m") == {
            "m_mean": 2.0,
            "m_min": 1.0,
            "m_max": 3.0,
        }

    def test_aggregate_of_nothing_is_empty(self):
        assert tracking.aggregate([None, None], "m") == {}

    def test_non_finite_values_are_dropped(self):
        assert tracking.aggregate([float("inf"), float("nan"), 2.0], "m")["m_mean"] == 2.0
