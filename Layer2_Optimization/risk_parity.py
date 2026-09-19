# Layer2_Optimization/risk_parity.py

"""Equal risk contribution ("risk parity") weights.

Solved through the convex log-barrier reformulation of Maillard, Roncalli and
Teiletche (2010):

    minimise  0.5 * w' S w  -  (1/n) * sum_i log(w_i),   w > 0

Its stationarity condition is ``(S w)_i * w_i = 1/n`` for every ``i`` -- exactly
the equal-risk-contribution condition -- and the objective is strictly convex, so
the solution is unique. Portfolio scale is pinned by the barrier rather than by a
sum-to-one constraint, so the result is normalised afterwards.

Two earlier formulations failed here and are worth recording:

1. Minimising ``sum((w_i (S w)_i - w'Sw/n)^2)`` on raw daily covariances puts the
   objective around 1e-10, below SLSQP's default ``ftol`` of 1e-6. The solver
   reported success without leaving its equal-weight starting point, returning
   near-equal weights whose risk contributions were far from equal.
2. Rescaling those terms to fractional contributions fixed the tolerance problem
   but left a non-convex objective on the simplex, which converged to a local
   optimum that zeroed one asset and split parity across the rest.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def risk_contributions(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """Each asset's share of total portfolio variance. Sums to 1."""
    weights = np.asarray(weights, dtype=float)
    contributions = weights * (np.asarray(cov_matrix, dtype=float) @ weights)
    total = contributions.sum()
    if total <= 0:
        return np.full(weights.size, 1.0 / weights.size)
    return contributions / total


def risk_parity_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """Long-only weights equalising each asset's fractional contribution to risk."""
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    n = cov_matrix.shape[0]
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)

    target = 1.0 / n

    # Work on a correlation-like scale so the barrier and the quadratic term are
    # comparable in magnitude regardless of the return frequency. Rescaling the
    # covariance by a positive constant leaves the normalised solution unchanged.
    scale = float(np.mean(np.diag(cov_matrix)))
    if not np.isfinite(scale) or scale <= 0:
        return np.repeat(target, n)
    scaled = cov_matrix / scale

    def objective(w: np.ndarray) -> float:
        return 0.5 * float(w @ scaled @ w) - target * float(np.sum(np.log(w)))

    def gradient(w: np.ndarray) -> np.ndarray:
        return scaled @ w - target / w

    vols = np.sqrt(np.diag(scaled))
    start = (1.0 / vols) / (1.0 / vols).sum() if np.all(vols > 0) else np.repeat(target, n)

    res = minimize(
        objective,
        start,
        jac=gradient,
        method="SLSQP",
        # Strictly positive: the log barrier is undefined at zero, and a genuine
        # ERC portfolio holds every asset.
        bounds=tuple((1e-9, None) for _ in range(n)),
        options={"ftol": 1e-14, "maxiter": 1000},
    )

    w = np.asarray(res.x, dtype=float)
    if not res.success or not np.isfinite(w).all() or w.sum() <= 0:
        import warnings

        from .mean_variance import SolverWarning

        warnings.warn(
            f"risk_parity: solver did not converge ({res.message}); using inverse-volatility "
            "weights, which are exact only for uncorrelated assets",
            SolverWarning,
            stacklevel=2,
        )
        return start

    return w / w.sum()
