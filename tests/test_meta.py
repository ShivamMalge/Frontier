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
    """Pins the exact strategy set, so an accidental addition or removal is caught."""
    strategies = client.get("/api/v1/meta/strategies").json()["strategies"]
    names = {s["name"] for s in strategies}
    assert names == {
        "Markowitz_MaxSharpe",
        "Markowitz_MinVar",
        "RiskParity",
        "GMV",
        "HRP",
        "HRP_CVaR",
        "Gerber_InvVar",
        "Gerber_HRP",
        "MinCVaR",
        "MinCDaR",
    }

    by_name = {s["name"]: s for s in strategies}
    # Only max-Sharpe consumes the return forecast; the other nine are risk-only,
    # so a better forecasting model cannot improve them.
    assert by_name["Markowitz_MaxSharpe"]["uses_expected_returns"] is True
    assert [n for n, s in by_name.items() if s["uses_expected_returns"]] == ["Markowitz_MaxSharpe"]
    # GMV permits shorts; its long-only twin does not.
    assert by_name["GMV"]["long_only"] is False
    assert by_name["Markowitz_MinVar"]["long_only"] is True
    # Strategies whose construction fixes every weight cannot honour extra bounds.
    assert by_name["RiskParity"]["respects_constraints"] is False
    assert by_name["Markowitz_MinVar"]["respects_constraints"] is True


def test_strategy_catalogue_reports_which_library_solves_each(client):
    strategies = {
        s["name"]: s["solver"] for s in client.get("/api/v1/meta/strategies").json()["strategies"]
    }
    assert strategies["Markowitz_MinVar"] == "cvxpy"
    assert strategies["HRP"] == "riskfolio"
