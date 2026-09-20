"""The Riskfolio-Lib strategies.

Each of these replaced a hand-rolled implementation that was either broken or not
the algorithm it claimed to be. The tests record what was wrong so a regression is
recognisable.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from frontier.optimization import riskfolio_strategies as rs


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(6)
    n = 8
    data = rng.normal(0.0005, 0.012, (900, n)) @ rng.normal(0, 1, (n, n)) * 0.4
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(n)])


ALL = [
    ("hrp", rs.hrp),
    ("hrp_cvar", rs.hrp_cvar),
    ("gerber_inverse_variance", rs.gerber_inverse_variance),
    ("gerber_hrp", rs.gerber_hrp),
    ("min_cvar", rs.min_cvar),
    ("min_cdar", rs.min_cdar),
]


@pytest.mark.parametrize(("name", "fn"), ALL)
class TestEveryStrategy:
    def test_weights_are_normalised_and_long_only(self, returns, name, fn):
        w = fn(returns)
        assert w.size == returns.shape[1]
        assert w.sum() == pytest.approx(1.0, abs=1e-8)
        assert (w >= -1e-9).all(), f"{name} produced a short position"

    def test_weights_are_finite(self, returns, name, fn):
        assert np.isfinite(fn(returns)).all()

    def test_is_deterministic(self, returns, name, fn):
        assert fn(returns) == pytest.approx(fn(returns), rel=1e-12)


class TestHRP:
    def test_holds_every_asset(self, returns):
        """Recursive bisection distributes across the whole tree."""
        assert (rs.hrp(returns) > 0).all()

    def test_terminates(self, returns):
        """The from-scratch version looped forever.

        Its quasi-diagonalisation thresholded on ``link.shape[0]`` (n-1) instead of
        the item count (n), so the highest-indexed asset was mistaken for a cluster
        and re-expanded without end.
        """
        assert rs.hrp(returns).size == returns.shape[1]

    def test_cvar_variant_differs_from_the_variance_variant(self, returns):
        """Measuring cluster risk by CVaR should change the allocation."""
        assert not np.allclose(rs.hrp(returns), rs.hrp_cvar(returns), atol=1e-4)


class TestGerber:
    def test_covariance_is_symmetric_and_positive_semidefinite(self, returns):
        """The hand-rolled version guaranteed neither.

        It thresholded on ``sign(r)`` and rescaled by ``(p - 0.5) / 0.5``, which is
        not the published statistic and can produce an indefinite matrix.
        """
        cov = rs.gerber_covariance(returns)
        assert cov == pytest.approx(cov.T)
        assert np.linalg.eigvalsh(cov).min() >= -1e-12

    def test_differs_from_the_sample_covariance(self, returns):
        """Thresholding small moves as noise should change the estimate."""
        assert not np.allclose(rs.gerber_covariance(returns), returns.cov().to_numpy(), atol=1e-8)

    def test_hrp_variant_uses_the_off_diagonals_unlike_inverse_variance(self, returns):
        """Inverse-variance weighting reads only the diagonal.

        Gerber_HRP clusters on the full matrix, so the two must disagree -- that
        difference is precisely the information Gerber_InvVar throws away.
        """
        assert not np.allclose(
            rs.gerber_inverse_variance(returns), rs.gerber_hrp(returns), atol=1e-3
        )

    def test_inverse_variance_depends_only_on_the_diagonal(self, returns):
        """Documents the limitation, rather than just asserting it in prose."""
        cov = rs.gerber_covariance(returns)
        expected = (1.0 / np.diag(cov)) / (1.0 / np.diag(cov)).sum()
        assert rs.gerber_inverse_variance(returns) == pytest.approx(expected, rel=1e-12)


class TestTailRisk:
    def test_min_cvar_concentrates_more_than_min_variance(self, returns):
        """Targeting only the left tail gives a different portfolio than variance."""
        from frontier.optimization.convex import min_variance

        assets = [str(c) for c in returns.columns]
        cvar = rs.min_cvar(returns)
        variance = min_variance(returns.cov().to_numpy(), assets)
        assert not np.allclose(cvar, variance, atol=1e-3)

    def test_min_cdar_differs_from_min_cvar(self, returns):
        """CDaR targets the path of losses; CVaR targets their distribution."""
        assert not np.allclose(rs.min_cdar(returns), rs.min_cvar(returns), atol=1e-3)

    def test_alpha_changes_the_cvar_solution(self, returns):
        assert not np.allclose(
            rs.min_cvar(returns, alpha=0.01), rs.min_cvar(returns, alpha=0.20), atol=1e-4
        )


@pytest.fixture
def near_singular_returns() -> pd.DataFrame:
    """Returns with a near-collinear pair, so the covariance has a tiny eigenvalue.

    Riskfolio's positive-definiteness check is ``all(eigenvalues >= 1e-6)`` -- an
    absolute threshold on a matrix whose variances are around 1e-4 -- so a genuinely
    small eigenvalue trips it. The base fixture does not reliably produce one, and a
    test of that behaviour must guarantee the condition it is testing.
    """
    rng = np.random.default_rng(11)
    base = rng.normal(0.0004, 0.011, (900, 5))
    # A sixth asset that is almost a copy of the first.
    duplicate = base[:, [0]] + rng.normal(0, 1e-5, (900, 1))
    data = np.hstack([base, duplicate])
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(6)])


class TestNoStdoutNoise:
    def test_riskfolio_pd_complaint_is_suppressed(self, near_singular_returns, capsys):
        """Riskfolio prints to stdout when eigenvalues fall below its 1e-6 floor.

        Daily return covariances legitimately have smaller eigenvalues than that, so
        the complaint is a false alarm on an absolute threshold. It must not reach a
        server's stdout.
        """
        returns = near_singular_returns
        assert np.linalg.eigvalsh(returns.cov().to_numpy()).min() < 1e-6
        rs.min_cvar(returns)
        rs.min_cdar(returns)
        assert capsys.readouterr().out == ""

    def test_the_pd_floor_does_not_change_tail_risk_weights(self, near_singular_returns):
        """Proves the fix is cosmetic: CVaR and CDaR never read the covariance."""
        import contextlib

        import riskfolio as rp

        returns = near_singular_returns

        def reference(measure: str) -> np.ndarray:
            with contextlib.redirect_stdout(io.StringIO()):
                portfolio = rp.Portfolio(returns=returns, alpha=rs.DEFAULT_ALPHA)
                portfolio.assets_stats(method_mu="hist", method_cov="hist")
                raw = portfolio.optimization(
                    model="Classic", rm=measure, obj="MinRisk", rf=0, hist=True
                )
            return np.asarray(raw, dtype=float).ravel()

        assert rs.min_cvar(returns) == pytest.approx(reference("CVaR"), abs=1e-8)
        assert rs.min_cdar(returns) == pytest.approx(reference("CDaR"), abs=1e-8)
