"""The backtest through the API, with real optimizers behind it."""

from __future__ import annotations

import time

import pytest

TICKERS = ["AAA", "BBB", "CCC", "DDD"]
WINDOW = {"start": "2018-01-01", "end": "2021-08-01"}
FAST = {"lookback": 120, "rebalance_every": 60, "cost_bps": 10.0}


def await_result(client, job_id: str, timeout: float = 300.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/pipeline/runs/{job_id}/result").json()
        if body["state"] not in {"queued", "running"}:
            return body
        time.sleep(0.05)
    pytest.fail(f"backtest {job_id} did not finish within {timeout}s")


def submit(client, **overrides) -> dict:
    payload = {"tickers": TICKERS, **WINDOW, **FAST, **overrides}
    response = client.post("/api/v1/backtest/runs", json=payload)
    assert response.status_code == 202, response.text
    body = await_result(client, response.json()["job_id"])
    assert body["state"] == "succeeded", body["message"]
    return body["result"]


def test_submitted_backtest_returns_a_job(client):
    response = client.post(
        "/api/v1/backtest/runs", json={"tickers": TICKERS, **WINDOW, **FAST}
    )
    assert response.status_code == 202
    assert response.json()["state"] in {"queued", "running"}
    assert "Location" in response.headers


def test_produces_out_of_sample_results_for_every_strategy(client):
    result = submit(client, strategies=["Markowitz_MinVar", "RiskParity", "HRP"])

    assert {r["strategy"] for r in result["results"]} == {
        "Markowitz_MinVar",
        "RiskParity",
        "HRP",
    }
    assert result["tickers"] == TICKERS
    assert result["observations"] > 0
    assert len(result["rebalance_dates"]) >= 2

    for row in result["results"]:
        assert -1.0 < row["annual_return"] < 2.0, row
        assert 0.0 < row["annual_volatility"] < 1.5, row
        assert -1.0 <= row["max_drawdown"] <= 0.0, row
        assert row["rebalances"] == len(result["rebalance_dates"])


def test_costs_are_charged_and_reported(client):
    result = submit(client, strategies=["Markowitz_MinVar"], cost_bps=100.0)
    row = result["results"][0]

    assert row["cost_drag"] > 0
    assert row["annual_return"] < row["gross_annual_return"]
    assert row["cost_drag"] == pytest.approx(
        row["gross_annual_return"] - row["annual_return"], rel=1e-6
    )
    assert row["total_cost"] > 0
    assert row["avg_turnover"] > 0


def test_zero_cost_removes_the_drag(client):
    result = submit(client, strategies=["Markowitz_MinVar"], cost_bps=0.0)
    row = result["results"][0]
    assert row["cost_drag"] == pytest.approx(0.0, abs=1e-12)
    assert row["total_cost"] == pytest.approx(0.0, abs=1e-12)


def test_net_growth_trails_gross_growth(client):
    result = submit(client, strategies=["Markowitz_MinVar"], cost_bps=100.0)
    net = result["cumulative_growth"]["data"][-1][0]
    gross = result["gross_cumulative_growth"]["data"][-1][0]
    assert net < gross


def test_turnover_frame_is_indexed_by_rebalance_date(client):
    result = submit(client, strategies=["RiskParity"])
    turnover = result["turnover"]
    assert turnover["index"] == result["rebalance_dates"]
    assert turnover["columns"] == ["RiskParity"]
    # Buying in from cash is the whole portfolio.
    assert turnover["data"][0][0] == pytest.approx(1.0, abs=1e-6)


def test_turnover_budget_is_respected_at_every_rebalance(client):
    """The per-strategy turnover path: each strategy trades from its own holdings."""
    result = submit(
        client,
        strategies=["Markowitz_MinVar"],
        constraints={
            "max_turnover": 0.15,
            "previous_weights": dict.fromkeys(TICKERS, 0.25),
        },
    )
    traded = [row[0] for row in result["turnover"]["data"]]
    # The opening trade is exempt -- nothing is held yet, so a cap would make
    # buying in impossible.
    assert all(t <= 0.15 + 1e-6 for t in traded[1:]), traded


def test_constraints_are_applied_during_the_backtest(client):
    result = submit(
        client, strategies=["Markowitz_MinVar"], constraints={"max_weight": 0.35}
    )
    assert result["results"][0]["rebalances"] >= 2
    assert not result["warnings"]


def test_lookback_longer_than_the_data_is_rejected(client):
    response = client.post(
        "/api/v1/backtest/runs",
        json={"tickers": TICKERS, **WINDOW, "lookback": 2000, "rebalance_every": 60},
    )
    job_id = response.json()["job_id"]
    body = await_result(client, job_id)
    assert body["state"] == "failed"
    assert "lookback" in body["message"]


def test_a_backtest_is_not_the_same_as_an_in_sample_optimisation(client):
    """The point of the whole exercise.

    In-sample weights are fitted on the data they are scored against; a
    walk-forward result is not, so the two should not coincide.
    """
    backtest = submit(client, strategies=["Markowitz_MinVar"])
    backtest_sharpe = backtest["results"][0]["sharpe"]

    returns = client.post(
        "/api/v1/market/returns", json={"tickers": TICKERS, **WINDOW}
    ).json()["returns"]
    weights = client.post(
        "/api/v1/portfolio/optimize",
        json={"returns": returns, "strategies": ["Markowitz_MinVar"]},
    ).json()["weights"]
    in_sample = client.post(
        "/api/v1/portfolio/performance", json={"returns": returns, "weights": weights}
    ).json()["performance"][0]["sharpe"]

    assert backtest_sharpe != pytest.approx(in_sample, rel=1e-3)
