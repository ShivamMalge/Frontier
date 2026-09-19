# Layer2_Optimization/optimizer_master.py

"""Single dispatch point for the optimizers.

Owns the strategy registry so that the CLI, the API service layer and the
Streamlit app all resolve strategy names the same way. Presentation metadata
(labels, descriptions) lives in the service layer; this module is numerical only.

Phase 4 swaps the individual optimizer implementations for cvxpy and
Riskfolio-Lib. The registry and this signature stay put.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from .gerber import gerber_covariance, gerber_inverse_var_weights
from .gmv import gmv_weights
from .hrp import hrp_allocation
from .mean_variance import maximize_sharpe, min_variance
from .risk_parity import risk_parity_weights

#: Registry order is the canonical presentation order.
STRATEGIES: tuple[str, ...] = (
    "Markowitz_MaxSharpe",
    "Markowitz_MinVar",
    "RiskParity",
    "GMV",
    "HRP",
    "Gerber_InvVar",
)


def _dispatch(
    mu: np.ndarray, cov: np.ndarray, returns: pd.DataFrame, rf: float
) -> dict[str, Callable[[], np.ndarray]]:
    return {
        "Markowitz_MaxSharpe": lambda: maximize_sharpe(mu, cov, rf=rf),
        "Markowitz_MinVar": lambda: min_variance(cov),
        "RiskParity": lambda: risk_parity_weights(cov),
        "GMV": lambda: gmv_weights(cov),
        "HRP": lambda: hrp_allocation(cov),
        "Gerber_InvVar": lambda: gerber_inverse_var_weights(
            gerber_covariance(returns)
        ).to_numpy(),
    }


def build_all_portfolios(
    returns: pd.DataFrame,
    strategies: list[str] | None = None,
    rf: float = 0.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(weights, warnings)`` for the requested strategies.

    ``returns`` must contain *returns*, not price levels. Weights are indexed by
    ticker with one column per strategy. A strategy that fails is reported in
    ``warnings`` and omitted from the frame rather than aborting the batch.
    """
    clean = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")
    selected = list(strategies) if strategies else list(STRATEGIES)

    mu = clean.mean().to_numpy()
    cov = clean.cov().to_numpy()
    available = _dispatch(mu, cov, clean, rf)

    columns: dict[str, np.ndarray] = {}
    messages: list[str] = []

    for name in selected:
        if name not in available:
            messages.append(f"{name}: unknown strategy, skipped")
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                weights = np.asarray(available[name](), dtype=float).ravel()
                messages.extend(f"{name}: {w.message}" for w in caught)
        except Exception as exc:
            messages.append(f"{name}: failed ({type(exc).__name__}: {exc})")
            continue

        if weights.size != clean.shape[1] or not np.isfinite(weights).all():
            messages.append(f"{name}: produced invalid weights, skipped")
            continue
        columns[name] = weights

    return pd.DataFrame(columns, index=clean.columns), messages
