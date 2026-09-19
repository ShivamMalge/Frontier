"""The walk-forward backtest engine.

The properties worth pinning down are the ones that make a backtest honest: that no
allocation sees the data it is scored on, that weights drift rather than being
pinned for free, and that trading costs something.

A trivial allocator is injected for most of these -- the engine's correctness has
nothing to do with which optimizer runs inside it, and equal weights make the
arithmetic checkable by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Layer3_Portfolio_Generation.backtest import (
    BacktestConfig,
    rebalance_schedule,
    run_backtest,
)

ASSETS = list("ABCDE")


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    index = pd.bdate_range("2018-01-01", periods=800)
    return pd.DataFrame(
        rng.normal(0.0004, 0.012, (800, len(ASSETS))), index=index, columns=ASSETS
    )


def equal_weight(window: pd.DataFrame, previous: dict) -> tuple[pd.DataFrame, list[str]]:
    n = window.shape[1]
    return pd.DataFrame({"EqualWeight": np.full(n, 1 / n)}, index=window.columns), []


def first_asset_only(window: pd.DataFrame, previous: dict) -> tuple[pd.DataFrame, list[str]]:
    weights = np.zeros(window.shape[1])
    weights[0] = 1.0
    return pd.DataFrame({"FirstOnly": weights}, index=window.columns), []


class TestSchedule:
    def test_starts_only_once_a_full_lookback_is_available(self):
        assert rebalance_schedule(800, 252, 21)[0] == 252

    def test_spacing_matches_the_request(self):
        schedule = rebalance_schedule(800, 252, 21)
        assert all(b - a == 21 for a, b in zip(schedule, schedule[1:], strict=False))

    def test_no_schedule_when_history_is_too_short(self):
        assert rebalance_schedule(200, 252, 21) == []

    def test_too_little_history_raises_rather_than_silently_doing_nothing(self, returns):
        with pytest.raises(ValueError, match="lookback"):
            run_backtest(returns.iloc[:100], equal_weight, BacktestConfig(lookback=252))


class TestNoLookahead:
    """The defect this engine exists to remove."""

    def test_evaluation_begins_at_the_first_rebalance(self, returns):
        result = run_backtest(returns, equal_weight, BacktestConfig(lookback=252))
        assert result.net_returns.index[0] == returns.index[252]
        assert len(result.net_returns) == len(returns) - 252

    def test_optimizer_only_ever_sees_data_before_the_day_it_allocates_for(self, returns):
        """Recorded window bounds must end strictly before each rebalance date."""
        seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        def recording(window, previous):
            seen.append((window.index[0], window.index[-1]))
            return equal_weight(window, previous)

        config = BacktestConfig(lookback=252, rebalance_every=21)
        result = run_backtest(returns, recording, config)

        assert len(seen) == len(result.rebalance_dates)
        for (_, window_end), rebalance_date in zip(seen, result.rebalance_dates, strict=True):
            assert window_end < rebalance_date

    def test_window_is_exactly_the_lookback_length(self, returns):
        lengths: list[int] = []

        def recording(window, previous):
            lengths.append(len(window))
            return equal_weight(window, previous)

        run_backtest(returns, recording, BacktestConfig(lookback=120))
        assert set(lengths) == {120}

    def test_future_data_cannot_change_past_results(self, returns):
        """Truncating the tail must leave earlier out-of-sample returns identical."""
        config = BacktestConfig(lookback=252, rebalance_every=21)
        full = run_backtest(returns, equal_weight, config).net_returns
        short = run_backtest(returns.iloc[:-100], equal_weight, config).net_returns

        overlap = short.index
        assert full.loc[overlap, "EqualWeight"].to_numpy() == pytest.approx(
            short["EqualWeight"].to_numpy(), rel=1e-12
        )


class TestWeightDrift:
    def test_weights_drift_with_prices_between_rebalances(self, returns):
        """Pinning weights daily would imply free trading and inflate returns.

        A single-asset portfolio is the clean test: holding only A must reproduce
        A's own returns exactly.
        """
        config = BacktestConfig(lookback=252, rebalance_every=21, cost_bps=0.0)
        result = run_backtest(returns, first_asset_only, config)
        expected = returns["A"].iloc[252:]
        assert result.gross_returns["FirstOnly"].to_numpy() == pytest.approx(
            expected.to_numpy(), rel=1e-9
        )

    def test_turnover_is_measured_against_drifted_weights(self, returns):
        """An equal-weight target still trades, because holdings drift away from it.

        Measuring against the previous *target* instead would report zero turnover
        and understate costs.
        """
        config = BacktestConfig(lookback=252, rebalance_every=21)
        result = run_backtest(returns, equal_weight, config)
        later = result.turnover["EqualWeight"].iloc[1:]
        assert (later > 0).all()
        # Small, though: a month of drift on equal weights is a few percent.
        assert later.max() < 0.5

    def test_first_rebalance_trades_the_whole_portfolio(self, returns):
        """Buying in from cash is 100% turnover by definition."""
        result = run_backtest(returns, equal_weight, BacktestConfig(lookback=252))
        assert result.turnover["EqualWeight"].iloc[0] == pytest.approx(1.0, abs=1e-9)


class TestCosts:
    def test_zero_cost_makes_net_equal_gross(self, returns):
        config = BacktestConfig(lookback=252, cost_bps=0.0)
        result = run_backtest(returns, equal_weight, config)
        assert result.net_returns.to_numpy() == pytest.approx(
            result.gross_returns.to_numpy(), rel=1e-12
        )

    def test_costs_reduce_net_returns(self, returns):
        config = BacktestConfig(lookback=252, cost_bps=50.0)
        result = run_backtest(returns, equal_weight, config)
        assert result.net_returns["EqualWeight"].sum() < result.gross_returns[
            "EqualWeight"
        ].sum()

    def test_cost_equals_turnover_times_the_rate(self, returns):
        config = BacktestConfig(lookback=252, cost_bps=25.0)
        result = run_backtest(returns, equal_weight, config)
        expected = result.turnover["EqualWeight"] * 0.0025
        assert result.costs["EqualWeight"].to_numpy() == pytest.approx(
            expected.to_numpy(), rel=1e-12
        )

    def test_cost_is_charged_once_per_rebalance_not_daily(self, returns):
        """A daily charge would overstate costs by the rebalance interval."""
        config = BacktestConfig(lookback=252, rebalance_every=21, cost_bps=100.0)
        result = run_backtest(returns, equal_weight, config)

        difference = result.gross_returns["EqualWeight"] - result.net_returns["EqualWeight"]
        charged_days = (difference.abs() > 1e-12).sum()
        assert charged_days == len(result.rebalance_dates)

    def test_higher_cost_hurts_more(self, returns):
        def drag(bps: float) -> float:
            config = BacktestConfig(lookback=252, cost_bps=bps)
            summary = run_backtest(returns, equal_weight, config).summary()
            return float(summary.loc["EqualWeight", "Cost Drag"])

        assert drag(100.0) > drag(10.0) > drag(0.0)
        assert drag(0.0) == pytest.approx(0.0, abs=1e-12)


class TestSummary:
    def test_reports_gross_net_and_drag_consistently(self, returns):
        config = BacktestConfig(lookback=252, cost_bps=20.0)
        summary = run_backtest(returns, equal_weight, config).summary()
        row = summary.loc["EqualWeight"]
        assert row["Cost Drag"] == pytest.approx(
            row["Gross Annual Return"] - row["Annual Return"], rel=1e-9
        )
        assert row["Rebalances"] == len(rebalance_schedule(len(returns), 252, 21))

    def test_cumulative_growth_starts_near_one_and_stays_positive(self, returns):
        result = run_backtest(returns, equal_weight, BacktestConfig(lookback=252))
        growth = result.cumulative()
        assert (growth > 0).all().all()
        assert float(growth.iloc[0, 0]) == pytest.approx(
            1.0 + result.net_returns.iloc[0, 0], rel=1e-12
        )

    def test_net_growth_trails_gross_growth(self, returns):
        config = BacktestConfig(lookback=252, cost_bps=50.0)
        result = run_backtest(returns, equal_weight, config)
        assert float(result.cumulative(net=True).iloc[-1, 0]) < float(
            result.cumulative(net=False).iloc[-1, 0]
        )


class TestRobustness:
    def test_a_failing_optimizer_is_reported_and_the_run_continues(self, returns):
        calls = {"n": 0}

        def flaky(window, previous):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("synthetic optimizer failure")
            return equal_weight(window, previous)

        result = run_backtest(returns, flaky, BacktestConfig(lookback=252))
        assert any("synthetic optimizer failure" in w for w in result.warnings)
        # The run still covers the whole out-of-sample span.
        assert len(result.net_returns) == len(returns) - 252

    def test_optimizer_failing_on_the_very_first_rebalance_raises(self, returns):
        def always_fails(window, previous):
            raise RuntimeError("no")

        with pytest.raises(ValueError, match="no strategies"):
            run_backtest(returns, always_fails, BacktestConfig(lookback=252))

    def test_single_asset_is_rejected(self, returns):
        with pytest.raises(ValueError, match="at least 2 assets"):
            run_backtest(returns[["A"]], equal_weight, BacktestConfig(lookback=252))

    def test_previous_weights_are_passed_to_the_optimizer(self, returns):
        """Needed for a turnover budget to mean anything."""
        seen: list[dict] = []

        def recording(window, previous):
            seen.append(previous)
            return equal_weight(window, previous)

        run_backtest(returns, recording, BacktestConfig(lookback=252))
        assert seen[0] == {}, "nothing is held before the first rebalance"
        assert "EqualWeight" in seen[1]
        assert set(seen[1]["EqualWeight"]) == set(ASSETS)
        assert sum(seen[1]["EqualWeight"].values()) == pytest.approx(1.0, abs=1e-9)
