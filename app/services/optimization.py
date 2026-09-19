"""Portfolio optimization.

Wraps the Layer 2 dispatch behind one entry point that takes a *returns* frame and
optional constraints, and returns a weights frame plus any non-fatal warnings.

Since Phase 4 the convex strategies are solved by cvxpy and the hierarchical,
robust-covariance and tail-risk strategies by Riskfolio-Lib.
"""

from __future__ import annotations

import pandas as pd

from app.errors import InsufficientDataError, OptimizationFailedError
from app.settings import get_settings
from Layer2_Optimization import convex
from Layer2_Optimization.constraints import DEFAULT, Constraints
from Layer2_Optimization.optimizer_master import STRATEGIES, build_all_portfolios

#: Display metadata for the strategy catalogue.
#:
#: ``uses_expected_returns`` is worth surfacing: only one of the ten strategies
#: consumes the return forecast at all. The other nine depend solely on the
#: covariance matrix or the return distribution, so a better forecasting model
#: cannot improve them.
STRATEGY_INFO: dict[str, dict[str, object]] = {
    "Markowitz_MaxSharpe": {
        "label": "Markowitz (max Sharpe)",
        "family": "mean-variance",
        "solver": "cvxpy",
        "long_only": True,
        "uses_expected_returns": True,
        "respects_constraints": True,
        "description": "Tangency portfolio, located by scanning the efficient frontier. "
        "Maximising a ratio is not convex, and the usual rescaling trick breaks once "
        "weight caps or turnover limits are present, so the frontier is traced instead.",
    },
    "Markowitz_MinVar": {
        "label": "Markowitz (minimum variance)",
        "family": "mean-variance",
        "solver": "cvxpy",
        "long_only": True,
        "uses_expected_returns": False,
        "respects_constraints": True,
        "description": "Minimises portfolio variance, long-only and fully invested.",
    },
    "RiskParity": {
        "label": "Risk parity",
        "family": "risk-based",
        "solver": "cvxpy",
        "long_only": True,
        "uses_expected_returns": False,
        "respects_constraints": False,
        "description": "Equalises each asset's contribution to portfolio variance, via the "
        "convex log-barrier formulation. Its stationarity condition fixes every weight, so "
        "additional bounds cannot be imposed.",
    },
    "GMV": {
        "label": "Global minimum variance",
        "family": "mean-variance",
        "solver": "cvxpy",
        "long_only": False,
        "uses_expected_returns": False,
        "respects_constraints": True,
        "description": "Minimum variance with short positions permitted. That permission is "
        "the only substantive difference from Markowitz (minimum variance).",
    },
    "HRP": {
        "label": "Hierarchical risk parity",
        "family": "hierarchical",
        "solver": "riskfolio",
        "long_only": True,
        "uses_expected_returns": False,
        "respects_constraints": False,
        "description": "Clusters assets by correlation distance, then allocates by recursive "
        "bisection on inverse cluster variance (Lopez de Prado, 2016).",
    },
    "HRP_CVaR": {
        "label": "Hierarchical risk parity (CVaR)",
        "family": "hierarchical",
        "solver": "riskfolio",
        "long_only": True,
        "uses_expected_returns": False,
        "respects_constraints": False,
        "description": "HRP with cluster risk measured by conditional value-at-risk instead "
        "of variance, so clustering responds to tail losses rather than overall dispersion.",
    },
    "Gerber_InvVar": {
        "label": "Gerber statistic (inverse variance)",
        "family": "robust-covariance",
        "solver": "riskfolio",
        "long_only": True,
        "uses_expected_returns": False,
        "respects_constraints": False,
        "description": "Inverse-variance weights on the Gerber covariance. Kept under its "
        "original name for continuity, but inverse-variance weighting reads only the "
        "diagonal, so the co-movement information is unused -- prefer Gerber_HRP.",
    },
    "Gerber_HRP": {
        "label": "Gerber statistic (HRP)",
        "family": "robust-covariance",
        "solver": "riskfolio",
        "long_only": True,
        "uses_expected_returns": False,
        "respects_constraints": False,
        "description": "HRP clustered on the Gerber covariance, which counts concordant and "
        "discordant moves beyond a volatility threshold and ignores small ones as noise. "
        "Unlike Gerber_InvVar this uses the whole matrix.",
    },
    "MinCVaR": {
        "label": "Minimum CVaR",
        "family": "tail-risk",
        "solver": "riskfolio",
        "long_only": True,
        "uses_expected_returns": False,
        "respects_constraints": True,
        "description": "Minimises the mean of the worst 5% of daily outcomes. Variance "
        "penalises upside and downside alike; this targets only the left tail.",
    },
    "MinCDaR": {
        "label": "Minimum CDaR",
        "family": "tail-risk",
        "solver": "riskfolio",
        "long_only": True,
        "uses_expected_returns": False,
        "respects_constraints": True,
        "description": "Minimises the mean of the worst 5% of drawdowns. The only strategy "
        "here that targets the path of losses rather than their distribution.",
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
    constraints: Constraints = DEFAULT,
) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(weights, warnings)``; weights are tickers x strategies.

    ``returns`` must contain returns, not price levels.
    """
    settings = get_settings()
    rf = settings.risk_free_rate if risk_free_rate is None else risk_free_rate

    clean = _validated(returns)
    try:
        weights, messages = build_all_portfolios(clean, strategies, rf, constraints)
    except ValueError as exc:
        # Contradictory constraints, reported before any solver ran.
        raise OptimizationFailedError(str(exc)) from exc

    if weights.empty:
        raise OptimizationFailedError(
            "every requested strategy failed", detail="; ".join(messages) or "no detail"
        )
    return weights, messages


def frontier(
    returns: pd.DataFrame,
    constraints: Constraints = DEFAULT,
    points: int = 40,
    risk_free_rate: float | None = None,
) -> list[convex.FrontierPoint]:
    """Trace the efficient frontier under ``constraints``.

    The project shipped a ``plot_efficient_frontier`` module from the start that
    plotted cumulative growth instead; this actually computes the curve.
    """
    settings = get_settings()
    rf = settings.risk_free_rate if risk_free_rate is None else risk_free_rate

    clean = _validated(returns)
    assets = [str(column) for column in clean.columns]

    problems = constraints.validate(assets)
    if problems:
        raise OptimizationFailedError("; ".join(problems))

    try:
        return convex.efficient_frontier(
            clean.mean().to_numpy(),
            clean.cov().to_numpy(),
            assets,
            constraints,
            points=points,
            periods_per_year=settings.trading_days_per_year,
            risk_free_rate=rf,
        )
    except convex.InfeasibleError as exc:
        raise OptimizationFailedError(str(exc)) from exc


def _validated(returns: pd.DataFrame) -> pd.DataFrame:
    clean = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if clean.shape[0] < 2:
        raise InsufficientDataError(
            f"need at least 2 complete return observations, got {clean.shape[0]}"
        )
    if clean.shape[1] < 2:
        raise InsufficientDataError(
            f"need at least 2 assets to optimize, got {clean.shape[1]}"
        )
    return clean
