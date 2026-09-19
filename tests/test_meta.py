"""Health and metadata endpoints."""

from __future__ import annotations


def test_health_reports_available_backends(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "naive" in body["forecast_backends"]


def test_universe_exposes_request_defaults(client):
    body = client.get("/api/v1/meta/universe").json()
    assert "AAPL" in body["tickers"]
    assert body["lookback_window"] == 60
    assert body["trading_days_per_year"] == 252


def test_strategy_catalogue_is_complete_and_honest(client):
    strategies = client.get("/api/v1/meta/strategies").json()["strategies"]
    names = {s["name"] for s in strategies}
    assert names == {
        "Markowitz_MaxSharpe",
        "Markowitz_MinVar",
        "RiskParity",
        "GMV",
        "HRP",
        "Gerber_InvVar",
    }
    by_name = {s["name"]: s for s in strategies}
    # Only max-Sharpe consumes the return forecast; the rest are risk-only.
    assert by_name["Markowitz_MaxSharpe"]["uses_expected_returns"] is True
    assert by_name["HRP"]["uses_expected_returns"] is False
    # GMV permits shorts; its long-only twin does not.
    assert by_name["GMV"]["long_only"] is False
    assert by_name["Markowitz_MinVar"]["long_only"] is True
