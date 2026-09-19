# Layer2_Optimization/convex.py

"""Convex portfolio optimizers, solved with cvxpy.

Replaces the hand-rolled ``scipy.optimize.minimize`` formulations. What that buys:

* **Correct answers, or an honest failure.** SLSQP returned ``res.x`` whether or
  not it converged, so a failed solve became plausible-looking garbage. cvxpy
  reports ``infeasible`` or ``unbounded`` explicitly.
* **Scale invariance.** The old risk-parity objective evaluated to ~1e-10 on daily
  covariances, under SLSQP's default tolerance, so it "converged" without moving.
  A conic solver does not have that failure mode.
* **Real constraints.** Sector caps, turnover limits and leverage bounds are
  linear inequalities here rather than bespoke callbacks.
* **An actual efficient frontier**, which the project plotted a function for but
  never computed.

Solvers: Clarabel first (interior-point, handles the exponential cone the
risk-parity barrier needs), then OSQP and SCS as fallbacks for the pure quadratic
programs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .constraints import DEFAULT, Constraints

logger = logging.getLogger(__name__)

#: Tried in order. Clarabel supports every cone used here.
SOLVER_ORDER = ("CLARABEL", "SCS", "OSQP")

#: Tightened tolerances. The defaults leave risk contributions off parity by ~1e-5,
#: which is visible when asserting that a risk-parity portfolio actually reaches it.
SOLVER_OPTIONS: dict[str, dict[str, float | int]] = {
    "CLARABEL": {"tol_gap_abs": 1e-12, "tol_gap_rel": 1e-12, "tol_feas": 1e-12},
    "SCS": {"eps": 1e-10, "max_iters": 20_000},
    "OSQP": {"eps_abs": 1e-10, "eps_rel": 1e-10, "max_iter": 50_000},
}


class InfeasibleError(ValueError):
    """No portfolio satisfies the requested constraints."""


@dataclass(frozen=True)
class FrontierPoint:
    expected_return: float
    volatility: float
    sharpe: float
    weights: np.ndarray


def _solve(problem, label: str) -> None:
    """Try each solver in turn; raise if none reaches an optimal solution."""
    import cvxpy as cp

    errors: list[str] = []
    for solver in SOLVER_ORDER:
        try:
            problem.solve(solver=solver, **SOLVER_OPTIONS.get(solver, {}))
        except Exception as exc:
            errors.append(f"{solver}: {type(exc).__name__}: {exc}")
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            if problem.status == cp.OPTIMAL_INACCURATE:
                logger.warning("%s: %s returned an inaccurate solution", label, solver)
            return
        errors.append(f"{solver}: {problem.status}")

    raise InfeasibleError(f"{label} could not be solved ({'; '.join(errors)})")


def _clean(weights: np.ndarray, constraints: Constraints) -> np.ndarray:
    """Zero out solver dust and renormalise to sum exactly 1."""
    w = np.asarray(weights, dtype=float).ravel()
    # Interior-point solvers leave ~1e-12 residuals where the answer is zero.
    floor = max(constraints.min_weight, 0.0)
    w[np.abs(w) < 1e-9] = floor if floor > 0 else 0.0
    total = w.sum()
    return w / total if abs(total) > 1e-12 else np.full(w.size, 1.0 / w.size)


def to_positive_definite(cov: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Symmetrise and nudge a covariance matrix onto the positive-definite cone.

    Sample covariances of noisy return data are routinely indefinite by a hair --
    enough to make a quadratic form non-convex and a solve fail, and enough for
    Riskfolio to refuse the matrix outright. The correction is the smallest
    diagonal shift that clears the smallest eigenvalue.
    """
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    smallest = float(np.linalg.eigvalsh(cov).min())
    if smallest < epsilon:
        cov = cov + np.eye(cov.shape[0]) * (epsilon - smallest)
    return cov


#: Retained as a private alias for readability inside this module.
_prepare = to_positive_definite


def min_variance(
    cov: np.ndarray, assets: list[str], constraints: Constraints = DEFAULT
) -> np.ndarray:
    """Minimum-variance portfolio subject to ``constraints``."""
    import cvxpy as cp

    cov = _prepare(cov)
    w = cp.Variable(cov.shape[0])
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))),
        constraints.cvxpy_constraints(w, assets),
    )
    _solve(problem, "min_variance")
    return _clean(w.value, constraints)


def max_return(
    mu: np.ndarray, assets: list[str], constraints: Constraints = DEFAULT
) -> np.ndarray:
    """Highest achievable expected return under ``constraints``."""
    import cvxpy as cp

    w = cp.Variable(len(mu))
    problem = cp.Problem(
        cp.Maximize(np.asarray(mu, dtype=float) @ w),
        constraints.cvxpy_constraints(w, assets),
    )
    _solve(problem, "max_return")
    return _clean(w.value, constraints)


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    assets: list[str],
    constraints: Constraints = DEFAULT,
    points: int = 40,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> list[FrontierPoint]:
    """Trace the efficient frontier by minimising variance at each target return.

    Returns annualised points, ordered by expected return. The endpoints are the
    minimum-variance and maximum-return portfolios, so the curve spans exactly the
    achievable range for these constraints.
    """
    import cvxpy as cp

    mu = np.asarray(mu, dtype=float)
    cov = _prepare(cov)

    lower = float(mu @ min_variance(cov, assets, constraints))
    upper = float(mu @ max_return(mu, assets, constraints))
    if upper < lower:
        lower, upper = upper, lower

    targets = np.linspace(lower, upper, max(points, 2))
    frontier: list[FrontierPoint] = []

    w = cp.Variable(len(mu))
    target = cp.Parameter()
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))),
        [*constraints.cvxpy_constraints(w, assets), mu @ w >= target],
    )

    for value in targets:
        target.value = float(value)
        try:
            _solve(problem, f"frontier@{value:.6g}")
        except InfeasibleError:
            # The very top of the range can be marginally infeasible from rounding.
            continue
        weights = _clean(w.value, constraints)
        frontier.append(_describe(weights, mu, cov, periods_per_year, risk_free_rate))

    if not frontier:
        raise InfeasibleError("no point on the frontier could be solved")
    return frontier


def _describe(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    periods_per_year: int,
    risk_free_rate: float,
) -> FrontierPoint:
    annual_return = float(mu @ weights) * periods_per_year
    annual_vol = float(np.sqrt(weights @ cov @ weights)) * np.sqrt(periods_per_year)
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0
    return FrontierPoint(annual_return, annual_vol, sharpe, weights)


def max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    assets: list[str],
    constraints: Constraints = DEFAULT,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    points: int = 60,
) -> np.ndarray:
    """Tangency portfolio, found by scanning the efficient frontier.

    Maximising a ratio is not convex. The usual fix -- minimise variance subject to
    ``(mu - rf)' w == 1`` and rescale -- only works when every other constraint is
    scale-invariant, which a ``max_weight`` cap or a turnover limit is not. Scanning
    the frontier respects arbitrary convex constraints, and the frontier is worth
    computing anyway.
    """
    frontier = efficient_frontier(
        mu, cov, assets, constraints, points, periods_per_year, risk_free_rate
    )
    return max(frontier, key=lambda point: point.sharpe).weights


def risk_parity(
    cov: np.ndarray, assets: list[str], constraints: Constraints = DEFAULT
) -> tuple[np.ndarray, list[str]]:
    """Equal risk contribution, via the convex log-barrier formulation.

    ``minimise 0.5 w' S w - (1/n) sum log(w)``, whose stationarity condition is
    ``w_i (S w)_i = 1/n`` for every asset -- exactly equal risk contributions.
    Strictly convex, so the solution is unique.

    Portfolio scale is pinned by the barrier rather than by a budget constraint, so
    the result is normalised afterwards. That also means box, leverage, group and
    turnover limits cannot be imposed here: they are not scale-invariant, and the
    equal-risk condition already determines every weight. Any such constraint is
    returned as a warning rather than silently ignored.
    """
    import cvxpy as cp

    cov = _prepare(cov)
    n = cov.shape[0]
    warnings: list[str] = []

    if constraints.max_weight < 1.0 or constraints.min_weight > 0.0:
        warnings.append(
            "RiskParity: weight bounds do not apply -- the equal-risk condition fixes "
            "every weight, so there is no freedom left to constrain"
        )
    if constraints.max_leverage is not None or constraints.group_caps:
        warnings.append("RiskParity: leverage and group limits do not apply")
    if constraints.max_turnover is not None:
        warnings.append("RiskParity: turnover limits do not apply")

    # Work on a correlation-like scale so the quadratic term and the barrier are
    # comparable regardless of return frequency; rescaling leaves the normalised
    # solution unchanged.
    scale = float(np.mean(np.diag(cov))) or 1.0
    scaled = cov / scale

    w = cp.Variable(n, pos=True)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(scaled)) - cp.sum(cp.log(w)) / n)
    )
    _solve(problem, "risk_parity")

    weights = np.asarray(w.value, dtype=float).ravel()
    total = weights.sum()
    if not np.isfinite(weights).all() or total <= 0:
        raise InfeasibleError("risk_parity produced non-finite weights")
    return weights / total, warnings


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Each asset's share of total portfolio variance. Sums to 1."""
    weights = np.asarray(weights, dtype=float)
    contributions = weights * (np.asarray(cov, dtype=float) @ weights)
    total = contributions.sum()
    if total <= 0:
        return np.full(weights.size, 1.0 / weights.size)
    return contributions / total
