"""The cvxpy optimizers and the constraint system.

Two things these pin down that the SLSQP versions could not: that a constraint is
actually honoured, and that an impossible request fails loudly instead of returning
plausible-looking weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from Layer2_Optimization import convex
from Layer2_Optimization.constraints import Constraints
from Layer2_Optimization.convex import (
    InfeasibleError,
    efficient_frontier,
    max_return,
    max_sharpe,
    min_variance,
    risk_contributions,
    risk_parity,
    to_positive_definite,
)


def sample(n: int = 8, periods: int = 1500, seed: int = 4):
    """Correlated returns plus the assets, mean and covariance."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.012, (periods, n)) @ rng.normal(0, 1, (n, n)) * 0.4
    return (
        [f"A{i}" for i in range(n)],
        returns.mean(axis=0),
        np.cov(returns, rowvar=False),
    )


class TestPositiveDefinite:
    def test_lifts_the_smallest_eigenvalue_above_the_floor(self):
        _, _, cov = sample()
        fixed = to_positive_definite(cov, epsilon=1e-6)
        assert np.linalg.eigvalsh(fixed).min() >= 1e-6 - 1e-15

    def test_symmetrises(self):
        cov = np.array([[1.0, 0.3], [0.1, 1.0]])
        fixed = to_positive_definite(cov)
        assert fixed == pytest.approx(fixed.T)

    def test_leaves_a_well_conditioned_matrix_essentially_alone(self):
        cov = np.eye(4) * 0.01
        assert to_positive_definite(cov) == pytest.approx(cov, abs=1e-9)


class TestMinVariance:
    def test_beats_equal_weight_on_variance(self):
        assets, _, cov = sample()
        w = min_variance(cov, assets)
        equal = np.full(len(assets), 1 / len(assets))
        assert w @ cov @ w <= equal @ cov @ equal + 1e-12

    def test_is_long_only_and_fully_invested(self):
        assets, _, cov = sample()
        w = min_variance(cov, assets)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert (w >= -1e-9).all()

    def test_shorting_lowers_achievable_variance(self):
        """GMV differs from long-only min-variance only by permitting shorts."""
        assets, _, cov = sample()
        long_only = min_variance(cov, assets)
        with_shorts = min_variance(cov, assets, Constraints(min_weight=-1.0, max_weight=1.0))
        assert with_shorts @ cov @ with_shorts <= long_only @ cov @ long_only + 1e-12
        assert with_shorts.min() < 0


class TestConstraints:
    def test_max_weight_cap_is_respected_exactly(self):
        assets, _, cov = sample()
        w = min_variance(cov, assets, Constraints(max_weight=0.20))
        assert w.max() <= 0.20 + 1e-6

    def test_min_weight_floor_forces_every_asset_in(self):
        assets, _, cov = sample()
        w = min_variance(cov, assets, Constraints(min_weight=0.05))
        assert w.min() >= 0.05 - 1e-6

    def test_leverage_cap_bounds_gross_exposure(self):
        """A 130/30 mandate: net 100%, gross at most 130%."""
        assets, _, cov = sample()
        w = min_variance(cov, assets, Constraints(min_weight=-0.5, max_leverage=1.3))
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.abs(w).sum() <= 1.3 + 1e-6

    def test_group_cap_is_respected(self):
        assets, _, cov = sample()
        group = assets[:3]
        w = min_variance(
            cov, assets, Constraints(group_caps={"first_three": (group, 0.25)})
        )
        held = sum(w[assets.index(a)] for a in group)
        assert held <= 0.25 + 1e-6

    def test_group_floor_is_respected(self):
        assets, _, cov = sample()
        group = assets[:2]
        w = min_variance(
            cov, assets, Constraints(group_floors={"first_two": (group, 0.40)})
        )
        held = sum(w[assets.index(a)] for a in group)
        assert held >= 0.40 - 1e-6

    def test_turnover_limit_keeps_the_portfolio_near_its_previous_state(self):
        assets, _, cov = sample()
        previous = dict.fromkeys(assets, 1.0 / len(assets))
        w = min_variance(
            cov,
            assets,
            Constraints(max_turnover=0.10, previous_weights=previous),
        )
        moved = sum(abs(w[i] - previous[a]) for i, a in enumerate(assets))
        assert moved <= 0.10 + 1e-6

    def test_tighter_turnover_moves_less(self):
        assets, _, cov = sample()
        previous = dict.fromkeys(assets, 1.0 / len(assets))

        def drift(limit: float) -> float:
            w = min_variance(
                cov, assets, Constraints(max_turnover=limit, previous_weights=previous)
            )
            return sum(abs(w[i] - previous[a]) for i, a in enumerate(assets))

        assert drift(0.05) <= drift(0.50) + 1e-9


class TestConstraintValidation:
    def test_max_weight_too_small_to_reach_full_investment_is_caught(self):
        assets = [f"A{i}" for i in range(10)]
        problems = Constraints(max_weight=0.05).validate(assets)
        assert any("cannot reach 1.0" in p for p in problems)

    def test_min_weight_too_large_is_caught(self):
        assets = [f"A{i}" for i in range(10)]
        problems = Constraints(min_weight=0.2).validate(assets)
        assert any("exceeds 1.0" in p for p in problems)

    def test_leverage_below_one_is_caught(self):
        problems = Constraints(max_leverage=0.8).validate(["A", "B"])
        assert any("below 1.0" in p for p in problems)

    def test_turnover_without_previous_weights_is_caught(self):
        problems = Constraints(max_turnover=0.2).validate(["A", "B"])
        assert any("previous_weights" in p for p in problems)

    def test_unknown_group_member_is_caught(self):
        problems = Constraints(group_caps={"x": (["NOPE"], 0.5)}).validate(["A", "B"])
        assert any("unknown assets" in p for p in problems)

    def test_a_valid_configuration_reports_nothing(self):
        assert Constraints(min_weight=0.0, max_weight=0.5).validate(["A", "B", "C"]) == []

    def test_genuinely_infeasible_constraints_raise_rather_than_guess(self):
        """SLSQP would have returned res.x regardless; cvxpy says infeasible."""
        assets, _, cov = sample(n=4)
        impossible = Constraints(
            min_weight=0.0,
            max_weight=1.0,
            group_caps={"all": (assets, 0.5)},
            group_floors={"same": (assets, 0.9)},
        )
        with pytest.raises((InfeasibleError, ValueError)):
            min_variance(cov, assets, impossible)


class TestRiskParity:
    @pytest.mark.parametrize("n", [2, 3, 8, 18])
    def test_risk_contributions_are_equal(self, n):
        assets, _, cov = sample(n=n)
        w, _ = risk_parity(cov, assets)
        assert risk_contributions(w, cov) == pytest.approx(np.full(n, 1 / n), abs=1e-8)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert (w > 0).all()

    def test_uncorrelated_assets_give_inverse_volatility_weights(self):
        vols = np.array([0.10, 0.20, 0.40])
        w, _ = risk_parity(np.diag(vols**2), ["A", "B", "C"])
        assert w == pytest.approx((1 / vols) / (1 / vols).sum(), abs=1e-6)

    def test_extra_constraints_produce_a_warning_rather_than_silence(self):
        """Its stationarity condition fixes every weight; bounds cannot apply."""
        assets, _, cov = sample()
        _, warnings = risk_parity(cov, assets, Constraints(max_weight=0.15))
        assert warnings
        assert any("do not apply" in w or "does not apply" in w for w in warnings)

    def test_no_warning_when_no_extra_constraints_were_asked_for(self):
        assets, _, cov = sample()
        _, warnings = risk_parity(cov, assets)
        assert warnings == []


class TestEfficientFrontier:
    def test_volatility_rises_monotonically_with_expected_return(self):
        assets, mu, cov = sample()
        points = efficient_frontier(mu, cov, assets, points=25)
        assert len(points) >= 20
        for earlier, later in zip(points, points[1:], strict=False):
            assert earlier.expected_return <= later.expected_return + 1e-12
            assert earlier.volatility <= later.volatility + 1e-6

    def test_endpoints_match_the_min_variance_and_max_return_portfolios(self):
        assets, mu, cov = sample()
        points = efficient_frontier(mu, cov, assets, points=30)

        lowest_vol = min(p.volatility for p in points)
        mv = min_variance(cov, assets)
        expected = float(np.sqrt(mv @ cov @ mv)) * np.sqrt(252)
        assert lowest_vol == pytest.approx(expected, rel=1e-4)

        mr = max_return(mu, assets)
        assert points[-1].expected_return == pytest.approx(float(mu @ mr) * 252, rel=1e-4)

    def test_every_point_is_fully_invested(self):
        assets, mu, cov = sample()
        for point in efficient_frontier(mu, cov, assets, points=15):
            assert point.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_constraints_shrink_the_attainable_range(self):
        assets, mu, cov = sample()
        wide = efficient_frontier(mu, cov, assets, points=20)
        capped = efficient_frontier(mu, cov, assets, Constraints(max_weight=0.2), points=20)
        assert capped[-1].expected_return <= wide[-1].expected_return + 1e-9


class TestMaxSharpe:
    def test_matches_the_best_point_on_the_frontier(self):
        assets, mu, cov = sample()
        w = max_sharpe(mu, cov, assets, points=60)
        best = max(efficient_frontier(mu, cov, assets, points=60), key=lambda p: p.sharpe)
        assert w == pytest.approx(best.weights, abs=1e-6)

    def test_beats_equal_weight_on_sharpe(self):
        assets, mu, cov = sample()
        w = max_sharpe(mu, cov, assets, points=60)
        equal = np.full(len(assets), 1 / len(assets))

        def sharpe(weights):
            return (mu @ weights) / np.sqrt(weights @ cov @ weights)

        assert sharpe(w) >= sharpe(equal)

    def test_respects_a_weight_cap(self):
        """The rescaling trick for max-Sharpe breaks under a box constraint.

        Scanning the frontier does not, which is why it is used here.
        """
        assets, mu, cov = sample()
        w = max_sharpe(mu, cov, assets, Constraints(max_weight=0.15), points=40)
        assert w.max() <= 0.15 + 1e-6
        assert w.sum() == pytest.approx(1.0, abs=1e-6)


class TestSolverFallback:
    def test_reports_failure_instead_of_returning_garbage(self, monkeypatch):
        """Every solver failing must raise, not hand back an arbitrary vector."""
        assets, _, cov = sample(n=4)
        monkeypatch.setattr(convex, "SOLVER_ORDER", ("NOT_A_SOLVER",))
        with pytest.raises(InfeasibleError, match="could not be solved"):
            min_variance(cov, assets)
