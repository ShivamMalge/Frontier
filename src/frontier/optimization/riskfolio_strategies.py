# src/frontier/optimization/riskfolio_strategies.py

"""Hierarchical, robust-covariance and tail-risk strategies via Riskfolio-Lib.

Replaces three hand-rolled implementations with a maintained library, and adds the
tail-risk measures that were missing:

* **HRP** -- was a from-scratch implementation with two defects that made it
  unusable: a quasi-diagonalisation threshold off by one, which looped forever, and
  cluster expansion through the ``pd.Series.append`` removed in pandas 2.0.
* **Gerber** -- was ``sign(r_i) * sign(r_j) > 0`` rescaled by ``(p - 0.5) / 0.5``,
  which is not the published statistic and produced a matrix with no PSD guarantee.
  Riskfolio implements the real thing, thresholding at +/- c * sigma with
  concordant and discordant counts, and returns a PSD matrix.
* **CVaR / CDaR** -- absent entirely. Variance penalises upside and downside
  alike; conditional value-at-risk and conditional drawdown-at-risk target the
  left tail, which is what an investor actually minds.

HERC is deliberately not offered: Riskfolio 7.3.0 raises ``TypeError`` from its own
``_hierarchical_recursive_bisection`` for that model regardless of arguments.
"""

from __future__ import annotations

import contextlib
import io
import logging
import warnings

import numpy as np
import pandas as pd

from .convex import to_positive_definite

logger = logging.getLogger(__name__)

#: Fraction of the worst outcomes that CVaR and CDaR average over.
DEFAULT_ALPHA = 0.05

#: Gerber threshold, in standard deviations. Moves smaller than this count as noise.
GERBER_THRESHOLD = 0.5

#: Smallest eigenvalue Riskfolio accepts (its `is_pos_def` default at the call sites
#: used here). CVaR and CDaR do not read the covariance at all, so clearing this
#: floor cannot change their weights -- a test asserts exactly that.
RISKFOLIO_PD_FLOOR = 1.1e-6


def _weights(frame) -> np.ndarray:
    """Riskfolio returns a single-column DataFrame; flatten it."""
    values = np.asarray(frame, dtype=float).ravel()
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("optimizer returned no usable weights")
    total = values.sum()
    return values / total if abs(total) > 1e-12 else np.full(values.size, 1.0 / values.size)


def _hierarchical(
    returns: pd.DataFrame,
    risk_measure: str = "MV",
    method_cov: str = "hist",
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:
    import riskfolio as rp

    with warnings.catch_warnings():
        # Riskfolio is chatty about pandas internals on some versions.
        warnings.simplefilter("ignore")
        portfolio = rp.HCPortfolio(returns=returns, alpha=alpha)
        result = portfolio.optimization(
            model="HRP",
            codependence="pearson",
            rm=risk_measure,
            method_cov=method_cov,
            linkage="single",
            leaf_order=True,
        )
    return _weights(result)


def hrp(returns: pd.DataFrame) -> np.ndarray:
    """Hierarchical risk parity on the sample covariance (Lopez de Prado, 2016)."""
    return _hierarchical(returns, risk_measure="MV")


def hrp_cvar(returns: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """HRP where cluster risk is measured by CVaR rather than variance."""
    return _hierarchical(returns, risk_measure="CVaR", alpha=alpha)


def gerber_covariance(returns: pd.DataFrame, threshold: float = GERBER_THRESHOLD):
    """The published Gerber statistic (GS2), as a covariance matrix.

    Counts pairs of returns that both exceed +/- ``threshold`` standard deviations
    in the same direction (concordant) or opposite directions (discordant), and
    ignores everything in between as noise. Unlike the previous hand-rolled
    version, the result is positive semi-definite.
    """
    import riskfolio as rp

    return np.asarray(rp.gerber_cov_stat2(returns, threshold=threshold), dtype=float)


def gerber_inverse_variance(
    returns: pd.DataFrame, threshold: float = GERBER_THRESHOLD
) -> np.ndarray:
    """Inverse-variance weights on the Gerber covariance.

    Retained under its original name so earlier results stay comparable. Note that
    inverse-variance weighting reads only the diagonal, so the co-movement
    information the Gerber statistic exists to capture is discarded -- use
    ``Gerber_MinVar`` or ``Gerber_HRP`` to actually use it.
    """
    cov = gerber_covariance(returns, threshold)
    inverse = 1.0 / np.diag(cov)
    return inverse / inverse.sum()


def gerber_hrp(returns: pd.DataFrame) -> np.ndarray:
    """HRP clustered on the Gerber covariance, which uses the full matrix."""
    return _hierarchical(returns, risk_measure="MV", method_cov="gerber2")


def _mean_risk(
    returns: pd.DataFrame,
    risk_measure: str,
    objective: str = "MinRisk",
    alpha: float = DEFAULT_ALPHA,
    risk_free_rate: float = 0.0,
) -> np.ndarray:
    import riskfolio as rp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        portfolio = rp.Portfolio(returns=returns, alpha=alpha)

        # Riskfolio tests positive-definiteness as `all(eigenvalues >= 1e-6)` --
        # an absolute threshold applied to a matrix whose variances are around
        # 1e-4 for daily returns, so a legitimately small eigenvalue trips it. It
        # then prints a complaint to stdout and carries on regardless. Swallow the
        # print during estimation, then supply a covariance that clears the
        # threshold so nothing downstream re-triggers it.
        with contextlib.redirect_stdout(io.StringIO()):
            portfolio.assets_stats(method_mu="hist", method_cov="hist")

        portfolio.cov = pd.DataFrame(
            to_positive_definite(portfolio.cov.to_numpy(), epsilon=RISKFOLIO_PD_FLOOR),
            index=portfolio.cov.index,
            columns=portfolio.cov.columns,
        )
        result = portfolio.optimization(
            model="Classic",
            rm=risk_measure,
            obj=objective,
            rf=risk_free_rate,
            hist=True,
        )
    if result is None:
        raise ValueError(f"{risk_measure}/{objective} was infeasible")
    return _weights(result)


def min_cvar(returns: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """Minimise conditional value-at-risk: the mean of the worst ``alpha`` of days."""
    return _mean_risk(returns, "CVaR", alpha=alpha)


def min_cdar(returns: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """Minimise conditional drawdown-at-risk: the mean of the worst drawdowns.

    The only strategy here that targets the *path* of losses rather than their
    distribution, which is what makes a portfolio survivable in practice.
    """
    return _mean_risk(returns, "CDaR", alpha=alpha)
