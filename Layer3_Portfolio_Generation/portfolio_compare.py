# Layer3_Portfolio_Generation/portfolio_compare.py

"""Performance attribution for portfolio return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.config import TRADING_DAYS_PER_YEAR


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline of the compounded series, as a negative fraction."""
    cumulative = (1.0 + returns).cumprod()
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
    return float(drawdown.min()) if len(drawdown) else 0.0


def downside_deviation(returns: pd.Series, target: float = 0.0) -> float:
    """Annualised downside deviation below ``target``.

    Shortfalls are averaged over *all* periods, not only the losing ones -- that
    is the standard definition. Dividing by the count of negative periods alone
    inflates the denominator and understates Sortino.
    """
    shortfall = np.minimum(returns - target, 0.0)
    if len(shortfall) == 0:
        return 0.0
    return float(np.sqrt(np.mean(shortfall**2)) * np.sqrt(TRADING_DAYS_PER_YEAR))


def compute_performance(returns_df: pd.DataFrame, rf: float = 0.0) -> pd.DataFrame:
    """Annualised risk/return statistics, one row per column of ``returns_df``.

    ``rf`` is an annualised rate expressed as a decimal.
    """
    rows: dict[str, dict[str, float]] = {}

    for column in returns_df.columns:
        series = returns_df[column].dropna()
        if series.empty:
            continue

        annual_return = float(series.mean() * TRADING_DAYS_PER_YEAR)
        annual_vol = float(series.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
        excess = annual_return - rf

        dd = downside_deviation(series)
        rows[str(column)] = {
            "Annual Return": annual_return,
            "Annual Vol": annual_vol,
            "Sharpe": float(excess / annual_vol) if annual_vol > 0 else 0.0,
            "Sortino": float(excess / dd) if dd > 0 else 0.0,
            "Max Drawdown": max_drawdown(series),
        }

    return pd.DataFrame(rows).T
