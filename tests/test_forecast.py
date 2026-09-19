"""Forecasting endpoint, including the price-vs-returns contract."""

from __future__ import annotations


def test_forecast_returns_both_prices_and_derived_returns(client, window):
    body = client.post(
        "/api/v1/forecast",
        json={"tickers": ["AAA", "BBB"], "backend": "naive", **window},
    ).json()

    assert body["backend"] == "naive"
    assert body["tickers"] == ["AAA", "BBB"]

    prices = [v for row in body["predicted_prices"]["data"] for v in row if v is not None]
    returns = [v for row in body["predicted_returns"]["data"] for v in row if v is not None]

    # The regression this guards: the original pipeline fed price levels into the
    # optimizers as though they were returns.
    assert min(prices) > 10.0, "predicted_prices should be price levels"
    assert max(abs(v) for v in returns) < 0.5, "predicted_returns should be returns"

    # Differencing costs exactly one observation.
    assert len(body["predicted_returns"]["index"]) == len(body["predicted_prices"]["index"]) - 1


def test_naive_backend_scores_exactly_one_on_mase(client, window):
    """The random walk is the MASE denominator, so it must score 1.0 by definition."""
    body = client.post(
        "/api/v1/forecast", json={"tickers": ["AAA"], "backend": "naive", **window}
    ).json()
    metrics = body["metrics"][0]
    assert metrics["mase_vs_naive"] == 1.0


def test_legacy_accuracy_metric_flatters_a_do_nothing_forecast(client, window):
    """Documents why the original 93% figure carried no information.

    The naive forecast makes no prediction at all beyond "no change", yet the
    legacy metric scores it above 95 and R^2 above 0.9. MASE, which compares
    against that same baseline, correctly reports 1.0.
    """
    body = client.post(
        "/api/v1/forecast", json={"tickers": ["AAA"], "backend": "naive", **window}
    ).json()
    metrics = body["metrics"][0]

    assert metrics["legacy_approximate_accuracy"] > 95.0
    assert metrics["r2"] > 0.9
    assert metrics["mase_vs_naive"] == 1.0
    # Null rather than 0.0: a flat forecast never calls a direction.
    assert metrics["directional_accuracy"] is None


def test_unknown_backend_is_rejected_at_validation(client, window):
    response = client.post(
        "/api/v1/forecast", json={"tickers": ["AAA"], "backend": "nonsense", **window}
    )
    assert response.status_code == 422


def test_failed_ticker_does_not_sink_the_request(client, monkeypatch, window):
    """One ticker raising must be reported, not abort the whole forecast."""
    import dataclasses

    from app.services import forecasting

    spec = forecasting._REGISTRY["naive"]

    def flaky(series, params):
        if series.name == "BBB":
            raise ValueError("synthetic failure")
        return spec.fn(series, params)

    monkeypatch.setitem(
        forecasting._REGISTRY, "naive", dataclasses.replace(spec, fn=flaky)
    )

    body = client.post(
        "/api/v1/forecast",
        json={"tickers": ["AAA", "BBB"], "backend": "naive", **window},
    ).json()

    assert body["tickers"] == ["AAA"]
    assert [f["ticker"] for f in body["failed"]] == ["BBB"]
    assert "synthetic failure" in body["failed"][0]["reason"]
