"""Unit tests for the optimizers in the numerical core.

These sit below the API and pin down the mathematics directly, including the
three defects found while building Phase 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from frontier.optimization.gmv import gmv_weights
from frontier.optimization.hrp import get_quasi_diag, hrp_allocation
from frontier.optimization.mean_variance import maximize_sharpe, min_variance
from frontier.optimization.risk_parity import risk_contributions, risk_parity_weights


def diagonal_cov(vols: np.ndarray) -> np.ndarray:
    return np.diag(vols**2)


def correlated_cov(n: int, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 1, (3000, n)) @ rng.normal(0, 1, (n, n)) * 0.01
    return np.cov(returns, rowvar=False)


class TestRiskParity:
    """Risk parity must equalise *risk*, which is not the same as equalising weights.

    The original objective -- squared deviation of raw risk contributions from
    ``portfolio_variance / n`` -- evaluates to about 1e-10 on daily covariances,
    below SLSQP's default ftol of 1e-6. The solver returned success without moving
    off equal weights, so the strategy silently degenerated into equal weighting.
    """

    def test_uncorrelated_assets_give_inverse_volatility_weights(self):
        vols = np.array([0.10, 0.20, 0.40])
        weights = risk_parity_weights(diagonal_cov(vols))
        expected = (1.0 / vols) / (1.0 / vols).sum()
        assert weights == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("n", [2, 3, 8, 18])
    def test_risk_contributions_are_equal(self, n: int):
        cov = correlated_cov(n)
        weights = risk_parity_weights(cov)
        contributions = risk_contributions(weights, cov)

        assert contributions == pytest.approx(np.full(n, 1.0 / n), abs=1e-6)
        assert weights.sum() == pytest.approx(1.0, abs=1e-9)
        # Every asset must be held; a zeroed asset is a local optimum, not parity.
        assert (weights > 1e-8).all()

    def test_unequal_volatilities_do_not_produce_equal_weights(self):
        """The precise symptom of the original bug."""
        cov = diagonal_cov(np.array([0.05, 0.20, 0.50]))
        weights = risk_parity_weights(cov)
        assert not np.allclose(weights, 1 / 3, atol=0.05)
        assert weights[0] > weights[1] > weights[2]


class TestHRP:
    """HRP previously either looped forever or crashed, depending on pandas version."""

    @pytest.mark.parametrize("n", [2, 3, 8, 18])
    def test_terminates_with_normalised_positive_weights(self, n: int):
        weights = hrp_allocation(correlated_cov(n))
        assert weights.shape == (n,)
        assert weights.sum() == pytest.approx(1.0, abs=1e-9)
        assert (weights > 0).all()

    def test_quasi_diag_returns_each_asset_exactly_once(self):
        """The ordering must be a permutation of the assets.

        The threshold was ``link.shape[0]`` (n-1) rather than the item count (n),
        so the highest-indexed asset was mistaken for a cluster and re-expanded
        forever.
        """
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform

        from frontier.optimization.hrp import correl_dist

        for n in (2, 5, 12):
            cov = correlated_cov(n)
            std = np.sqrt(np.diag(cov))
            corr = cov / np.outer(std, std)
            link = linkage(squareform(correl_dist(corr), checks=False), method="single")
            assert sorted(get_quasi_diag(link)) == list(range(n))


class TestMeanVariance:
    def test_min_variance_beats_equal_weight_on_variance(self):
        cov = correlated_cov(8)
        weights = min_variance(cov)
        equal = np.repeat(1 / 8, 8)
        assert weights @ cov @ weights <= equal @ cov @ equal + 1e-12

    def test_min_variance_is_long_only_and_gmv_need_not_be(self):
        cov = correlated_cov(8)
        assert (min_variance(cov) >= -1e-8).all()
        assert gmv_weights(cov).sum() == pytest.approx(1.0, abs=1e-9)

    def test_max_sharpe_prefers_the_better_reward_per_unit_risk(self):
        cov = diagonal_cov(np.array([0.2, 0.2]))
        # Identical risk, second asset has twice the expected return.
        weights = maximize_sharpe(np.array([0.001, 0.002]), cov)
        assert weights[1] > weights[0]

    def test_weights_sum_to_one(self):
        cov = correlated_cov(6)
        mu = np.linspace(0.0001, 0.001, 6)
        for weights in (maximize_sharpe(mu, cov), min_variance(cov), gmv_weights(cov)):
            assert weights.sum() == pytest.approx(1.0, abs=1e-6)
