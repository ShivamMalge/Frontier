"""Dataset construction: the alignment and leakage guarantees.

These are the tests that matter most in Phase 3. A forecasting pipeline with an
off-by-one or a scaler fitted on the wrong slice produces impressive metrics and
no information, which is exactly what the original project suffered from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Layer1_LSTM.datasets import build_split
from Layer1_LSTM.scaling import MinMax, Standardiser
from Layer1_Preprocessing.feature_engineering import build_features, feature_names
from Layer1_Preprocessing.synthetic import synthetic_prices


@pytest.fixture
def prices() -> pd.Series:
    series = synthetic_prices(["AAA"], "2015-01-01", "2023-01-01")["AAA"]
    series.name = "AAA"
    return series


class TestScalers:
    @pytest.mark.parametrize("cls", [MinMax, Standardiser])
    def test_round_trip_is_exact(self, cls, prices):
        values = prices.to_numpy()
        scaler = cls.fit(values[:500])
        restored = np.asarray(scaler.inverse(scaler.transform(values)))
        assert restored == pytest.approx(values, abs=1e-9)

    def test_minmax_is_fitted_on_the_training_slice_only(self, prices):
        """The leak this replaces: fitting on the full series before splitting.

        Test values must be free to fall outside [0, 1]; if they do not, the
        scaler saw the future.
        """
        values = prices.to_numpy().reshape(-1, 1)
        split = int(np.ceil(len(values) * 2 / 3))
        scaler = MinMax.fit(values[:split])
        scaled = np.asarray(scaler.transform(values))

        assert scaled[:split].min() == pytest.approx(0.0, abs=1e-9)
        assert scaled[:split].max() == pytest.approx(1.0, abs=1e-9)
        # A trending series leaves the training range out of sample.
        assert scaled[split:].min() < 0.0 or scaled[split:].max() > 1.0

    @pytest.mark.parametrize("cls", [MinMax, Standardiser])
    def test_constant_series_does_not_divide_by_zero(self, cls):
        scaler = cls.fit(np.full(100, 7.0))
        assert np.isfinite(scaler.transform(np.full(10, 7.0))).all()


class TestFeatures:
    def test_columns_match_the_declared_order(self, prices):
        frame = build_features(prices)
        assert list(frame.columns) == feature_names()

    def test_no_feature_uses_future_information(self, prices):
        """Truncating the series must not change features on the rows that remain.

        Any feature peeking forward would shift when later data disappears.
        """
        full = build_features(prices)
        truncated = build_features(prices.iloc[:-50])

        overlap = truncated.index
        pd.testing.assert_frame_equal(
            full.loc[overlap], truncated, check_exact=False, atol=1e-12
        )

    def test_warm_up_rows_are_nan_not_fabricated(self, prices):
        frame = build_features(prices)
        # The longest lookback is 63 days, so early rows cannot be complete.
        assert frame.iloc[:60].isna().any(axis=1).all()
        assert not frame.iloc[200:].isna().any(axis=1).any()

    def test_rsi_stays_in_range(self, prices):
        rsi = build_features(prices)["rsi_14"].dropna()
        assert rsi.between(0.0, 1.0).all()


class TestSplitAlignment:
    @pytest.mark.parametrize("window", [None, 30])
    def test_a_zero_return_prediction_reproduces_the_naive_forecast(self, prices, window):
        """The single most important alignment check.

        Predicting "no change" must rebuild exactly the previous close. If the
        target is off by one, this fails.
        """
        split = build_split(prices, train_split=2 / 3, target="return", window=window)
        zero_scaled = split.target_scaler.transform(np.zeros(split.y_test.size))
        rebuilt = split.rebuild_prices(zero_scaled, "return")
        assert rebuilt == pytest.approx(split.test_previous_price, rel=1e-12)

    @pytest.mark.parametrize("window", [None, 30])
    def test_perfect_predictions_rebuild_the_actual_prices(self, prices, window):
        """Feeding the true targets back must recover the realised series."""
        split = build_split(prices, train_split=2 / 3, target="return", window=window)
        rebuilt = split.rebuild_prices(split.y_test, "return")
        assert rebuilt == pytest.approx(split.test_actual_price, rel=1e-9)

    def test_price_target_rebuilds_directly(self, prices):
        split = build_split(prices, train_split=2 / 3, target="price", window=None)
        rebuilt = split.rebuild_prices(split.y_test, "price")
        assert rebuilt == pytest.approx(split.test_actual_price, rel=1e-9)

    @pytest.mark.parametrize("window", [None, 30])
    def test_previous_price_precedes_the_actual_price(self, prices, window):
        """test_previous_price[i] must be the close before test_actual_price[i]."""
        split = build_split(prices, train_split=2 / 3, window=window)
        # Consecutive actuals line up with the next row's "previous".
        assert split.test_previous_price[1:] == pytest.approx(split.test_actual_price[:-1])

    def test_target_scaler_never_sees_test_targets(self, prices):
        """Fitted on training targets only, so training stats must reproduce it."""
        split = build_split(prices, train_split=2 / 3, window=None)
        refitted = Standardiser.fit(split.target_scaler.inverse(split.y_train))
        assert refitted.mean == pytest.approx(split.target_scaler.mean, rel=1e-9)
        assert refitted.scale == pytest.approx(split.target_scaler.scale, rel=1e-9)

    def test_sequence_and_tabular_forms_cover_the_same_test_window(self, prices):
        tabular = build_split(prices, train_split=2 / 3, window=None)
        sequence = build_split(prices, train_split=2 / 3, window=30)
        assert list(sequence.test_dates) == list(tabular.test_dates)
        assert sequence.test_actual_price == pytest.approx(tabular.test_actual_price)

    def test_sequences_have_the_requested_shape(self, prices):
        split = build_split(prices, train_split=2 / 3, window=45)
        assert split.x_train.shape[1:] == (45, len(feature_names()))

    def test_too_little_data_is_rejected_clearly(self):
        short = synthetic_prices(["AAA"], "2022-01-01", "2022-04-01")["AAA"]
        short.name = "AAA"
        with pytest.raises(ValueError, match="training rows|no usable rows|sequences"):
            build_split(short, train_split=2 / 3, window=60)
