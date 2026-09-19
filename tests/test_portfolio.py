"""Optimization and performance endpoints."""

from __future__ import annotations

import pytest


def test_optimize_returns_normalised_weights_for_every_strategy(client, returns_frame):
    body = client.post(
        "/api/v1/portfolio/optimize", json={"returns": returns_frame}
    ).json()

    assert set(body["weights"]["columns"]) == {
        "Markowitz_MaxSharpe",
        "Markowitz_MinVar",
        "RiskParity",
        "GMV",
        "HRP",
        "Gerber_InvVar",
    }
    assert body["weights"]["index"] == ["AAA", "BBB", "CCC", "DDD"]

    columns = body["weights"]["columns"]
    for position, name in enumerate(columns):
        total = sum(row[position] for row in body["weights"]["data"])
        assert total == pytest.approx(1.0, abs=1e-6), f"{name} weights must sum to 1"


def test_hrp_terminates_and_differentiates_assets(client, returns_frame):
    """Guards the two historical HRP defects: an infinite loop and a pandas crash.

    ``get_quasi_diag`` thresholded on ``link.shape[0]`` instead of the item count,
    so the highest-numbered asset was re-expanded forever; and the expansion used
    ``pd.Series.append``, removed in pandas 2.0.
    """
    body = client.post(
        "/api/v1/portfolio/optimize",
        json={"returns": returns_frame, "strategies": ["HRP"]},
    ).json()

    weights = [row[0] for row in body["weights"]["data"]]
    assert sum(weights) == pytest.approx(1.0, abs=1e-9)
    assert all(w > 0 for w in weights)
    assert len({round(w, 9) for w in weights}) > 1, "HRP must differentiate assets"


def test_optimize_rejects_price_levels(client, returns_frame):
    """The guard that makes the original bug impossible to reintroduce silently."""
    prices = {
        **returns_frame,
        "data": [[100.0 + i for i, _ in enumerate(row)] for row in returns_frame["data"]],
    }
    response = client.post("/api/v1/portfolio/optimize", json={"returns": prices})
    assert response.status_code == 422
    assert "returns" in response.text


def test_single_asset_is_rejected(client, returns_frame):
    one = {
        "index": returns_frame["index"],
        "columns": ["AAA"],
        "data": [[row[0]] for row in returns_frame["data"]],
    }
    response = client.post("/api/v1/portfolio/optimize", json={"returns": one})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "insufficient_data"


def test_ragged_frame_is_rejected(client):
    bad = {"index": ["2020-01-01"], "columns": ["A", "B"], "data": [[0.01]]}
    response = client.post("/api/v1/portfolio/optimize", json={"returns": bad})
    assert response.status_code == 422


def test_performance_produces_plausible_annualised_statistics(client, returns_frame):
    weights = client.post(
        "/api/v1/portfolio/optimize", json={"returns": returns_frame}
    ).json()["weights"]

    body = client.post(
        "/api/v1/portfolio/performance",
        json={"returns": returns_frame, "weights": weights},
    ).json()

    assert len(body["performance"]) == 6
    for row in body["performance"]:
        # The original code annualised price levels and produced returns in the
        # tens of thousands; these bounds would have caught it immediately.
        assert -1.0 < row["annual_return"] < 2.0, row
        assert 0.0 < row["annual_volatility"] < 1.0, row
        assert -1.0 <= row["max_drawdown"] <= 0.0, row

    growth = body["cumulative_growth"]
    assert growth["columns"] == weights["columns"]
    assert all(v is not None and v > 0 for v in growth["data"][-1])


def test_performance_aligns_on_tickers_not_position(client, returns_frame):
    """Weights given in a different order must still be matched by label."""
    weights = client.post(
        "/api/v1/portfolio/optimize", json={"returns": returns_frame}
    ).json()["weights"]

    order = [3, 1, 0, 2]
    shuffled = {
        "index": [weights["index"][i] for i in order],
        "columns": weights["columns"],
        "data": [weights["data"][i] for i in order],
    }

    baseline = client.post(
        "/api/v1/portfolio/performance",
        json={"returns": returns_frame, "weights": weights},
    ).json()["performance"]
    reordered = client.post(
        "/api/v1/portfolio/performance",
        json={"returns": returns_frame, "weights": shuffled},
    ).json()["performance"]

    # Re-associating the same sum shifts the last bits, so compare numerically.
    assert [r["strategy"] for r in baseline] == [r["strategy"] for r in reordered]
    for left, right in zip(baseline, reordered, strict=True):
        for field in ("annual_return", "annual_volatility", "sharpe", "sortino", "max_drawdown"):
            assert left[field] == pytest.approx(right[field], rel=1e-12), (field, left, right)
