"""The Polars data pipeline.

The point of these is equivalence, not novelty: the Polars implementation must
agree with the pandas one it replaced, to floating-point tolerance and including
which rows are NaN during the warm-up. A faster pipeline that quietly computes
something slightly different would be worse than the slow one.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from app.schemas.common import Frame
from Layer1_Preprocessing.feature_engineering import build_features, feature_names
from Layer1_Preprocessing.features_polars import build_features_all, build_features_lazy
from Layer1_Preprocessing.frames import (
    long_to_wide,
    pandas_wide_to_polars,
    polars_wide_to_pandas,
    wide_to_long,
)
from Layer1_Preprocessing.preprocessing import (
    compute_returns,
    compute_returns_long,
    compute_returns_polars,
)
from Layer1_Preprocessing.synthetic import synthetic_prices

TICKERS = ["AAA", "BBB", "CCC"]


@pytest.fixture
def pandas_prices():
    return synthetic_prices(TICKERS, "2018-01-01", "2023-01-01")


@pytest.fixture
def wide(pandas_prices):
    return pandas_wide_to_polars(pandas_prices)


@pytest.fixture
def long(wide):
    return wide_to_long(wide)


class TestConversions:
    def test_pandas_to_polars_moves_the_index_into_a_column(self, pandas_prices, wide):
        assert wide.columns == ["date", *TICKERS]
        assert wide.height == len(pandas_prices)

    def test_round_trip_preserves_values(self, pandas_prices, wide):
        restored = polars_wide_to_pandas(wide)
        assert restored.to_numpy() == pytest.approx(pandas_prices.to_numpy())
        assert list(restored.columns) == list(pandas_prices.columns)

    def test_wide_to_long_and_back(self, wide):
        restored = long_to_wide(wide_to_long(wide))
        assert restored.columns == wide.columns
        assert restored.select(TICKERS).to_numpy() == pytest.approx(wide.select(TICKERS).to_numpy())

    def test_long_format_has_one_row_per_date_and_ticker(self, wide, long):
        assert long.columns == ["date", "ticker", "price"]
        assert long.height == wide.height * len(TICKERS)
        assert sorted(long["ticker"].unique().to_list()) == sorted(TICKERS)


class TestFeatureEquivalence:
    """The Polars features must match the pandas implementation exactly."""

    def test_all_columns_present_in_declared_order(self, long):
        result = build_features_all(long)
        assert [c for c in result.columns if c in feature_names()] == feature_names()

    @pytest.mark.parametrize("ticker", TICKERS)
    def test_values_match_pandas(self, pandas_prices, long, ticker):
        reference = build_features(pandas_prices[ticker])
        produced = build_features_all(long).filter(pl.col("ticker") == ticker).sort("date")

        for name in feature_names():
            expected = reference[name].to_numpy().astype(float)
            actual = produced[name].to_numpy().astype(float)
            finite = ~(np.isnan(expected) | np.isnan(actual))
            assert actual[finite] == pytest.approx(expected[finite], abs=1e-9, rel=1e-9), name

    @pytest.mark.parametrize("ticker", TICKERS)
    def test_warm_up_nan_pattern_matches_pandas(self, pandas_prices, long, ticker):
        """Where features are undefined matters as much as their values.

        A shifted warm-up would mean the two pipelines disagree about which rows are
        usable, which changes the train/test split.
        """
        reference = build_features(pandas_prices[ticker])
        produced = build_features_all(long).filter(pl.col("ticker") == ticker).sort("date")

        for name in feature_names():
            expected = np.isnan(reference[name].to_numpy().astype(float))
            actual = np.isnan(produced[name].to_numpy().astype(float))
            assert np.array_equal(expected, actual), name

    def test_tickers_do_not_bleed_into_each_other(self, long):
        """Window functions must partition by ticker.

        Without ``over("ticker")`` a rolling window would run off the end of one
        series and into the next -- the kind of silent error that is invisible in
        aggregate metrics.
        """
        featured = build_features_all(long)
        for ticker in TICKERS:
            rows = featured.filter(pl.col("ticker") == ticker).sort("date")
            # The longest lookback is 63 days, so the first 63 rows of *each*
            # ticker must be incomplete, not just the first ticker's.
            assert rows["vol_63"].head(62).is_null().all()
            assert rows["vol_63"].tail(1).is_not_null().all()

    def test_intermediate_columns_are_dropped(self, long):
        result = build_features_all(long)
        assert not [c for c in result.columns if c.startswith("__")]

    def test_lazy_and_eager_agree(self, long):
        eager = build_features_all(long)
        lazy = build_features_lazy(long.lazy()).collect()
        assert lazy.equals(eager)

    def test_lag_count_is_configurable(self, long):
        result = build_features_all(long, lags=3)
        assert "ret_lag_3" in result.columns
        assert "ret_lag_4" not in result.columns


class TestReturns:
    def test_wide_polars_returns_match_pandas(self, pandas_prices, wide):
        expected = compute_returns(pandas_prices)
        produced = compute_returns_polars(wide)

        assert produced.height == len(expected)
        assert produced.drop("date").to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-12)

    def test_long_returns_are_grouped_per_ticker(self, long):
        result = compute_returns_long(long)
        assert "return" in result.columns
        # One row lost per ticker to differencing, not one row overall.
        assert result.height == long.height - len(TICKERS)

    def test_long_returns_match_wide_returns(self, wide, long):
        wide_returns = compute_returns_polars(wide)
        long_returns = compute_returns_long(long)

        for ticker in TICKERS:
            expected = wide_returns[ticker].to_numpy()
            actual = (
                long_returns.filter(pl.col("ticker") == ticker).sort("date")["return"].to_numpy()
            )
            assert actual == pytest.approx(expected, abs=1e-12)


class TestFrameFromPolars:
    def test_serialises_without_a_pandas_hop(self, wide):
        frame = Frame.from_polars(wide)
        assert frame.columns == TICKERS
        assert len(frame.index) == wide.height
        assert frame.index[0] == str(wide["date"][0])[:10]

    def test_nan_and_infinity_become_null(self):
        frame = pl.DataFrame(
            {
                "date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
                "X": [1.0, float("inf")],
                "Y": [float("nan"), 2.0],
            }
        )
        assert Frame.from_polars(frame).data == [[1.0, None], [None, 2.0]]

    def test_agrees_with_from_pandas(self, pandas_prices, wide):
        assert Frame.from_polars(wide).model_dump() == Frame.from_pandas(pandas_prices).model_dump()

    def test_missing_index_column_is_rejected(self, wide):
        with pytest.raises(ValueError, match="index column"):
            Frame.from_polars(wide, index_column="nope")


class TestDatasetsUseThePolarsPath:
    def test_universe_builder_returns_one_split_per_ticker(self, pandas_prices):
        from Layer1_LSTM.datasets import build_splits

        splits, failures = build_splits(pandas_prices, train_split=2 / 3, window=None)
        assert sorted(splits) == sorted(TICKERS)
        assert failures == []
        assert all(s.n_features == len(feature_names()) for s in splits.values())

    def test_single_series_helper_agrees_with_the_universe_builder(self, pandas_prices):
        from Layer1_LSTM.datasets import build_split, build_splits

        splits, _ = build_splits(pandas_prices, train_split=2 / 3, window=None)
        series = pandas_prices["AAA"]
        series.name = "AAA"
        single = build_split(series, train_split=2 / 3, window=None)

        assert single.x_train == pytest.approx(splits["AAA"].x_train, abs=1e-12)
        assert single.y_test == pytest.approx(splits["AAA"].y_test, abs=1e-12)
        assert list(single.test_dates) == list(splits["AAA"].test_dates)

    def test_a_bad_ticker_is_reported_not_fatal(self, pandas_prices):
        from Layer1_LSTM.datasets import build_splits

        short = pandas_prices.copy()
        short.loc[short.index[10:], "BBB"] = np.nan

        splits, failures = build_splits(short, train_split=2 / 3, window=None)
        assert "AAA" in splits
        assert [name for name, _ in failures] == ["BBB"]
