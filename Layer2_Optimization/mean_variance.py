# Layer2_Optimization/mean_variance.py

import numpy as np
from scipy.optimize import minimize

from utils.config import TRADING_DAYS_PER_YEAR


def weight_bounds(n, lb=0.0, ub=1.0):
    return tuple((lb, ub) for _ in range(n))

def constraint_sum_to_one():
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

def maximize_sharpe(mean_returns, cov_matrix, rf=0.0):
    n = len(mean_returns)
    bounds = weight_bounds(n, 0.0, 1.0)
    start = np.repeat(1.0 / n, n)

    def neg_sharpe(w):
        ret = np.dot(w, mean_returns) * TRADING_DAYS_PER_YEAR
        vol = np.sqrt(w.T @ (cov_matrix * TRADING_DAYS_PER_YEAR) @ w)
        if vol == 0:
            return 1e9
        return - (ret - rf) / vol

    cons = (constraint_sum_to_one(),)
    res = minimize(neg_sharpe, start, method="SLSQP", bounds=bounds, constraints=cons)
    return _checked(res, start, "maximize_sharpe")

def min_variance(cov_matrix):
    n = cov_matrix.shape[0]
    bounds = weight_bounds(n, 0.0, 1.0)
    start = np.repeat(1.0 / n, n)

    def vol(w):
        return np.sqrt(w.T @ cov_matrix @ w)

    cons = (constraint_sum_to_one(),)
    res = minimize(vol, start, method="SLSQP", bounds=bounds, constraints=cons)
    return _checked(res, start, "min_variance")


class SolverWarning(UserWarning):
    """Raised as a warning when SLSQP reports non-convergence."""


def _checked(res, fallback, label):
    """Return the solution, or fall back to equal weights if SLSQP failed.

    Previously ``res.x`` was returned unconditionally, so a failed solve became
    silent garbage weights. Phase 4 replaces SLSQP with cvxpy, which reports
    infeasibility explicitly.
    """
    import warnings

    if not res.success:
        warnings.warn(
            f"{label}: SLSQP did not converge ({res.message}); using equal weights",
            SolverWarning,
            stacklevel=2,
        )
        return fallback.copy()
    return res.x
