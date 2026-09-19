"""Portfolio construction and performance attribution."""

from __future__ import annotations

import pandas as pd

from app.errors import InsufficientDataError
from app.settings import get_settings
from Layer3_Portfolio_Generation.portfolio_builder import build_portfolio_returns
from Layer3_Portfolio_Generation.portfolio_compare import compute_performance
from Layer3_Portfolio_Generation.portfolio_selector import choose_portfolio_by_risk


def evaluate(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    risk_free_rate: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(performance, portfolio_returns, cumulative_growth)``."""
    settings = get_settings()
    rf = settings.risk_free_rate if risk_free_rate is None else risk_free_rate

    shared = [ticker for ticker in weights.index if ticker in returns.columns]
    if not shared:
        raise InsufficientDataError(
            "no overlap between the weight index and the return columns",
            weight_tickers=",".join(map(str, weights.index)),
            return_tickers=",".join(map(str, returns.columns)),
        )

    aligned_returns = returns[shared].dropna(axis=0, how="any")
    aligned_weights = weights.loc[shared]
    if aligned_returns.empty:
        raise InsufficientDataError("no complete return observations after alignment")

    portfolio_returns = build_portfolio_returns(aligned_returns, aligned_weights)
    performance = compute_performance(portfolio_returns, rf=rf)
    cumulative = (1.0 + portfolio_returns).cumprod()
    return performance, portfolio_returns, cumulative


def select(performance: pd.DataFrame, risk_tolerance: float) -> str:
    return str(choose_portfolio_by_risk(performance, risk_tolerance))
