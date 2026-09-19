"""Portfolio optimization.

Wraps the Layer 2 optimizers behind one entry point that takes a *returns* frame
and returns a weights frame. Solver non-convergence is collected as warnings
rather than swallowed.

Phase 4 replaces the SLSQP-based optimizers with cvxpy (Clarabel/OSQP) and HRP /
Gerber / CVaR with Riskfolio-Lib, behind this same function signature.
"""

from __future__ import annotations

import pandas as pd

from app.errors import InsufficientDataError, OptimizationFailedError
from app.settings import get_settings
from Layer2_Optimization.optimizer_master import STRATEGIES, build_all_portfolios

#: Display metadata for the strategy catalogue. ``uses_expected_returns`` is
#: worth surfacing: four of the six optimizers ignore the return forecast
#: entirely and depend only on the covariance matrix.
STRATEGY_INFO: dict[str, dict[str, object]] = {
    "Markowitz_MaxSharpe": {
        "label": "Markowitz (max Sharpe)",
        "family": "mean-variance",
        "long_only": True,
        "uses_expected_returns": True,
        "description": "Maximises the Sharpe ratio of annualised expected return over "
        "annualised volatility, long-only and fully invested.",
    },
    "Markowitz_MinVar": {
        "label": "Markowitz (minimum variance)",
        "family": "mean-variance",
        "long_only": True,
        "uses_expected_returns": False,
        "description": "Minimises portfolio variance subject to long-only, fully-invested "
        "constraints. Mathematically the long-only counterpart of GMV.",
    },
    "RiskParity": {
        "label": "Risk parity",
        "family": "risk-based",
        "long_only": True,
        "uses_expected_returns": False,
        "description": "Equalises each asset's contribution to total portfolio risk.",
    },
    "GMV": {
        "label": "Global minimum variance",
        "family": "mean-variance",
        "long_only": False,
        "uses_expected_returns": False,
        "description": "Closed-form minimum-variance solution via the pseudo-inverse of the "
        "covariance matrix. Shorts are permitted, which is the only substantive "
        "difference from Markowitz (minimum variance).",
    },
    "HRP": {
        "label": "Hierarchical risk parity",
        "family": "risk-based",
        "long_only": True,
        "uses_expected_returns": False,
        "description": "Clusters assets by correlation distance, then allocates by recursive "
        "bisection weighted on inverse cluster variance (Lopez de Prado, 2016).",
    },
    "Gerber_InvVar": {
        "label": "Gerber statistic (inverse variance)",
        "family": "robust-covariance",
        "long_only": True,
        "uses_expected_returns": False,
        "description": "Inverse-variance weights on a co-movement matrix built from paired "
        "return signs. Note that inverse-variance weighting reads only the "
        "diagonal, so the off-diagonal co-movement information is unused.",
    },
}

ALL_STRATEGIES: tuple[str, ...] = STRATEGIES

assert set(STRATEGY_INFO) == set(STRATEGIES), (
    "STRATEGY_INFO and the optimizer registry have diverged"
)


def optimize(
    returns: pd.DataFrame,
    strategies: list[str] | None = None,
    risk_free_rate: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(weights, warnings)``; weights are tickers x strategies.

    ``returns`` must contain returns, not price levels. Validation and error
    translation live here; the dispatch itself is in
    :func:`Layer2_Optimization.optimizer_master.build_all_portfolios`.
    """
    settings = get_settings()
    rf = settings.risk_free_rate if risk_free_rate is None else risk_free_rate

    clean = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if clean.shape[0] < 2:
        raise InsufficientDataError(
            f"need at least 2 complete return observations, got {clean.shape[0]}"
        )
    if clean.shape[1] < 2:
        raise InsufficientDataError(
            f"need at least 2 assets to optimize, got {clean.shape[1]}"
        )

    weights, messages = build_all_portfolios(clean, strategies, rf)
    if weights.empty:
        raise OptimizationFailedError(
            "every requested strategy failed", detail="; ".join(messages) or "no detail"
        )
    return weights, messages
