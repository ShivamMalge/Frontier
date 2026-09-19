# Layer2_Optimization/optimizer_master.py

"""Single dispatch point for the optimizers.

Owns the strategy registry so the CLI, the API service layer and the Streamlit app
resolve names identically. Numerical only -- presentation metadata lives in the
service layer.

Since Phase 4 the convex strategies are solved by cvxpy (``convex.py``) and the
hierarchical, robust-covariance and tail-risk strategies by Riskfolio-Lib
(``riskfolio_strategies.py``). The hand-rolled SLSQP modules are retained for
comparison but are no longer what the API calls.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.config import TRADING_DAYS_PER_YEAR

from . import convex, riskfolio_strategies
from .constraints import DEFAULT, Constraints

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyResult:
    weights: np.ndarray
    warnings: list[str]


#: Registry order is the canonical presentation order.
STRATEGIES: tuple[str, ...] = (
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
)


def _dispatch(
    returns: pd.DataFrame,
    mu: np.ndarray,
    cov: np.ndarray,
    assets: list[str],
    constraints: Constraints,
    rf: float,
) -> dict[str, Callable[[], StrategyResult]]:
    def plain(weights: np.ndarray) -> StrategyResult:
        return StrategyResult(weights, [])

    def unconstrained(name: str, weights: np.ndarray) -> StrategyResult:
        """Wrap a strategy whose definition leaves no room for extra constraints."""
        notes: list[str] = []
        if _has_extra_constraints(constraints):
            notes.append(
                f"{name}: weight, leverage, group and turnover limits are not applied -- "
                "this strategy's construction determines every weight"
            )
        return StrategyResult(weights, notes)

    # GMV is minimum variance with shorting permitted; that is the only difference
    # from Markowitz_MinVar, so it overrides the caller's bounds by design.
    gmv_constraints = Constraints(
        min_weight=min(constraints.min_weight, -1.0),
        max_weight=max(constraints.max_weight, 1.0),
        group_caps=constraints.group_caps,
        group_floors=constraints.group_floors,
    )

    return {
        "Markowitz_MaxSharpe": lambda: plain(
            convex.max_sharpe(
                mu, cov, assets, constraints, rf, TRADING_DAYS_PER_YEAR
            )
        ),
        "Markowitz_MinVar": lambda: plain(convex.min_variance(cov, assets, constraints)),
        "RiskParity": lambda: StrategyResult(*convex.risk_parity(cov, assets, constraints)),
        "GMV": lambda: plain(convex.min_variance(cov, assets, gmv_constraints)),
        "HRP": lambda: unconstrained("HRP", riskfolio_strategies.hrp(returns)),
        "HRP_CVaR": lambda: unconstrained("HRP_CVaR", riskfolio_strategies.hrp_cvar(returns)),
        "Gerber_InvVar": lambda: unconstrained(
            "Gerber_InvVar", riskfolio_strategies.gerber_inverse_variance(returns)
        ),
        "Gerber_HRP": lambda: unconstrained(
            "Gerber_HRP", riskfolio_strategies.gerber_hrp(returns)
        ),
        "MinCVaR": lambda: plain(riskfolio_strategies.min_cvar(returns)),
        "MinCDaR": lambda: plain(riskfolio_strategies.min_cdar(returns)),
    }


def _has_extra_constraints(constraints: Constraints) -> bool:
    return bool(
        constraints.min_weight > 0.0
        or constraints.max_weight < 1.0
        or constraints.max_leverage is not None
        or constraints.group_caps
        or constraints.group_floors
        or constraints.max_turnover is not None
    )


def build_all_portfolios(
    returns: pd.DataFrame,
    strategies: list[str] | None = None,
    rf: float = 0.0,
    constraints: Constraints = DEFAULT,
) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(weights, warnings)`` for the requested strategies.

    ``returns`` must contain *returns*, not price levels. Weights are indexed by
    ticker with one column per strategy. A strategy that fails is reported in
    ``warnings`` and omitted rather than aborting the batch.
    """
    clean = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")
    selected = list(strategies) if strategies else list(STRATEGIES)
    assets = [str(column) for column in clean.columns]

    messages: list[str] = []
    problems = constraints.validate(assets)
    if problems:
        # Contradictory constraints produce a clearer message here than a generic
        # solver infeasibility would.
        raise ValueError("; ".join(problems))

    mu = clean.mean().to_numpy()
    cov = clean.cov().to_numpy()
    available = _dispatch(clean, mu, cov, assets, constraints, rf)

    columns: dict[str, np.ndarray] = {}

    for name in selected:
        if name not in available:
            messages.append(f"{name}: unknown strategy, skipped")
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = available[name]()
                messages.extend(f"{name}: {item.message}" for item in caught)
        except Exception as exc:
            messages.append(f"{name}: failed ({type(exc).__name__}: {exc})")
            continue

        weights = np.asarray(result.weights, dtype=float).ravel()
        if weights.size != len(assets) or not np.isfinite(weights).all():
            messages.append(f"{name}: produced invalid weights, skipped")
            continue

        columns[name] = weights
        messages.extend(result.warnings)

    return pd.DataFrame(columns, index=clean.columns), messages
