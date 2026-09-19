"""Optimization and performance endpoints."""

from __future__ import annotations

from itertools import pairwise

import pytest


def test_optimize_returns_normalised_weights_for_every_strategy(client, returns_frame):
    body = client.post("/api/v1/portfolio/optimize", json={"returns": returns_frame}).json()

    from app.services.optimization import ALL_STRATEGIES

    assert set(body["weights"]["columns"]) == set(ALL_STRATEGIES)
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
    weights = client.post("/api/v1/portfolio/optimize", json={"returns": returns_frame}).json()[
        "weights"
    ]

    body = client.post(
        "/api/v1/portfolio/performance",
        json={"returns": returns_frame, "weights": weights},
    ).json()

    from app.services.optimization import ALL_STRATEGIES

    assert len(body["performance"]) == len(ALL_STRATEGIES)
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
    weights = client.post("/api/v1/portfolio/optimize", json={"returns": returns_frame}).json()[
        "weights"
    ]

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


class TestConstraintsThroughTheApi:
    """Constraints are the practical payoff of moving from SLSQP to cvxpy."""

    def test_weight_cap_is_applied(self, client, returns_frame):
        body = client.post(
            "/api/v1/portfolio/optimize",
            json={
                "returns": returns_frame,
                "strategies": ["Markowitz_MinVar"],
                "constraints": {"max_weight": 0.30},
            },
        ).json()
        weights = [row[0] for row in body["weights"]["data"]]
        assert max(weights) <= 0.30 + 1e-6
        assert sum(weights) == pytest.approx(1.0, abs=1e-6)

    def test_group_cap_is_applied(self, client, returns_frame):
        body = client.post(
            "/api/v1/portfolio/optimize",
            json={
                "returns": returns_frame,
                "strategies": ["Markowitz_MinVar"],
                "constraints": {"group_caps": {"pair": [["AAA", "BBB"], 0.25]}},
            },
        ).json()
        index = body["weights"]["index"]
        weights = {t: row[0] for t, row in zip(index, body["weights"]["data"], strict=True)}
        assert weights["AAA"] + weights["BBB"] <= 0.25 + 1e-6

    def test_turnover_limit_is_applied(self, client, returns_frame):
        previous = {"AAA": 0.25, "BBB": 0.25, "CCC": 0.25, "DDD": 0.25}
        body = client.post(
            "/api/v1/portfolio/optimize",
            json={
                "returns": returns_frame,
                "strategies": ["Markowitz_MinVar"],
                "constraints": {"max_turnover": 0.10, "previous_weights": previous},
            },
        ).json()
        index = body["weights"]["index"]
        weights = {t: row[0] for t, row in zip(index, body["weights"]["data"], strict=True)}
        moved = sum(abs(weights[t] - previous[t]) for t in previous)
        assert moved <= 0.10 + 1e-6

    def test_short_selling_mandate(self, client, returns_frame):
        body = client.post(
            "/api/v1/portfolio/optimize",
            json={
                "returns": returns_frame,
                "strategies": ["Markowitz_MinVar"],
                "constraints": {"min_weight": -0.3, "max_leverage": 1.3},
            },
        ).json()
        weights = [row[0] for row in body["weights"]["data"]]
        assert sum(weights) == pytest.approx(1.0, abs=1e-6)
        assert sum(abs(w) for w in weights) <= 1.3 + 1e-6

    def test_impossible_constraints_are_rejected_before_solving(self, client, returns_frame):
        """A clear message beats a generic solver infeasibility."""
        response = client.post(
            "/api/v1/portfolio/optimize",
            json={"returns": returns_frame, "constraints": {"max_weight": 0.1}},
        )
        assert response.status_code == 422
        assert "cannot reach 1.0" in response.text

    def test_turnover_without_previous_weights_is_rejected(self, client, returns_frame):
        response = client.post(
            "/api/v1/portfolio/optimize",
            json={"returns": returns_frame, "constraints": {"max_turnover": 0.2}},
        )
        assert response.status_code == 422

    def test_strategies_that_cannot_honour_constraints_say_so(self, client, returns_frame):
        """Silence would be worse: the caller would assume the cap was applied."""
        body = client.post(
            "/api/v1/portfolio/optimize",
            json={
                "returns": returns_frame,
                "strategies": ["RiskParity", "HRP"],
                "constraints": {"max_weight": 0.30},
            },
        ).json()
        assert body["warnings"]
        assert any("RiskParity" in w for w in body["warnings"])
        assert any("HRP" in w for w in body["warnings"])


class TestEfficientFrontier:
    """The project shipped a plot_efficient_frontier module that never computed one."""

    def test_returns_an_upward_sloping_curve(self, client, returns_frame):
        body = client.post(
            "/api/v1/portfolio/frontier", json={"returns": returns_frame, "points": 20}
        ).json()

        points = body["points"]
        assert len(points) >= 15
        for earlier, later in pairwise(points):
            assert earlier["expected_return"] <= later["expected_return"] + 1e-9
            assert earlier["volatility"] <= later["volatility"] + 1e-6

    def test_every_point_is_fully_invested(self, client, returns_frame):
        body = client.post(
            "/api/v1/portfolio/frontier", json={"returns": returns_frame, "points": 10}
        ).json()
        for point in body["points"]:
            assert sum(point["weights"].values()) == pytest.approx(1.0, abs=1e-6)
            assert set(point["weights"]) == set(body["tickers"])

    def test_tangency_index_marks_the_best_sharpe(self, client, returns_frame):
        body = client.post(
            "/api/v1/portfolio/frontier", json={"returns": returns_frame, "points": 30}
        ).json()
        sharpes = [p["sharpe"] for p in body["points"]]
        assert body["tangency_index"] == sharpes.index(max(sharpes))

    def test_tangency_point_matches_the_max_sharpe_strategy(self, client, returns_frame):
        """Markowitz_MaxSharpe is defined as this point, so they must agree."""
        frontier = client.post(
            "/api/v1/portfolio/frontier", json={"returns": returns_frame, "points": 60}
        ).json()
        tangency = frontier["points"][frontier["tangency_index"]]["weights"]

        optimised = client.post(
            "/api/v1/portfolio/optimize",
            json={"returns": returns_frame, "strategies": ["Markowitz_MaxSharpe"]},
        ).json()
        index = optimised["weights"]["index"]
        weights = {t: row[0] for t, row in zip(index, optimised["weights"]["data"], strict=True)}

        for ticker, weight in weights.items():
            assert weight == pytest.approx(tangency[ticker], abs=1e-4)

    def test_constraints_narrow_the_frontier(self, client, returns_frame):
        def top_return(payload: dict) -> float:
            body = client.post("/api/v1/portfolio/frontier", json=payload).json()
            return body["points"][-1]["expected_return"]

        wide = top_return({"returns": returns_frame, "points": 20})
        capped = top_return(
            {"returns": returns_frame, "points": 20, "constraints": {"max_weight": 0.3}}
        )
        assert capped <= wide + 1e-9
