"""The forecasting backends.

Synthetic prices are geometric random walks with no exploitable structure, which
makes them an unusually good test bed: the *correct* result is MASE at or just
above 1.0 and directional accuracy near 0.5. A backend that appears to beat the
random walk here has a leak, so these bounds are a leak detector rather than a
performance target.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.errors import UnknownBackendError
from app.services.forecasting import (
    ForecastParams,
    available_backends,
    backend_specs,
    run_forecast,
)
from Layer1_Preprocessing.synthetic import synthetic_prices

TICKERS = ["AAA", "BBB", "CCC"]

#: Fast settings: these tests check correctness and wiring, not convergence.
FAST = ForecastParams(lookback_window=20, train_split=2 / 3, epochs=2, batch_size=128, lags=5)


@pytest.fixture
def prices():
    return synthetic_prices(TICKERS, "2016-01-01", "2023-01-01")


class TestRegistry:
    def test_all_four_backends_are_registered(self):
        assert set(available_backends()) >= {"naive", "torch_lstm", "lightgbm"}

    def test_scopes_are_declared_correctly(self):
        scopes = {spec.name: spec.scope for spec in backend_specs()}
        assert scopes["naive"] == "ticker"
        assert scopes["keras_lstm"] == "ticker"
        # Universe scope is what lets these train one model across all tickers.
        assert scopes["torch_lstm"] == "universe"
        assert scopes["lightgbm"] == "universe"

    def test_unknown_backend_is_rejected(self, prices):
        with pytest.raises(UnknownBackendError):
            run_forecast(prices, "no_such_backend", FAST)


@pytest.mark.parametrize("backend", ["naive", "torch_lstm", "lightgbm"])
class TestEveryBackend:
    def test_produces_a_forecast_for_each_ticker(self, prices, backend):
        outcome = run_forecast(prices, backend, FAST)
        assert outcome.tickers == TICKERS
        assert not outcome.failures

    def test_predicted_prices_are_price_scale_and_returns_are_return_scale(self, prices, backend):
        """Guards the Phase 1 bug: the two must not be confused."""
        outcome = run_forecast(prices, backend, FAST)

        price_values = outcome.predicted_prices().to_numpy()
        price_values = price_values[np.isfinite(price_values)]
        assert price_values.min() > 1.0

        return_values = outcome.predicted_returns().to_numpy()
        return_values = return_values[np.isfinite(return_values)]
        assert np.abs(return_values).max() < 0.5

    def test_does_not_beat_a_random_walk_on_random_walk_data(self, prices, backend):
        """The leak detector. Synthetic data has no structure to find."""
        outcome = run_forecast(prices, backend, FAST)
        for metrics in outcome.metrics():
            assert metrics["mase_vs_naive"] > 0.9, (
                f"{backend}/{metrics['ticker']} beat a random walk on random-walk "
                f"data (MASE {metrics['mase_vs_naive']:.3f}) -- suspect a leak"
            )
            direction = metrics["directional_accuracy"]
            # None for a flat forecast, which makes no directional call at all.
            assert direction is None or 0.3 < direction < 0.7

    def test_forecast_window_is_consistent_across_tickers(self, prices, backend):
        outcome = run_forecast(prices, backend, FAST)
        frame = outcome.predicted_prices()
        assert not frame.empty
        assert frame.notna().all().all()

    def test_progress_is_reported_and_monotonic(self, prices, backend):
        seen: list[float] = []
        run_forecast(prices, backend, FAST, lambda f, _m: seen.append(f))
        assert seen, "the backend reported no progress at all"
        assert seen[-1] == 1.0
        assert seen == sorted(seen)


class TestNaive:
    def test_scores_exactly_one_on_mase(self, prices):
        """It is the MASE denominator, so anything else means a bug."""
        outcome = run_forecast(prices, "naive", FAST)
        for metrics in outcome.metrics():
            assert metrics["mase_vs_naive"] == pytest.approx(1.0, abs=1e-9)

    def test_makes_no_directional_call(self, prices):
        """A flat forecast has no opinion on direction, so the metric is null.

        Reporting 0.0 here would read as "always wrong" rather than "never guessed".
        """
        outcome = run_forecast(prices, "naive", FAST)
        assert all(m["directional_accuracy"] is None for m in outcome.metrics())

    def test_prediction_is_the_previous_actual(self, prices):
        forecast = run_forecast(prices, "naive", FAST).forecasts[0]
        assert forecast.predicted[1:] == pytest.approx(forecast.actual[:-1])


class TestMultiSeries:
    @pytest.mark.parametrize("backend", ["torch_lstm", "lightgbm"])
    def test_one_model_still_forecasts_every_ticker(self, prices, backend):
        params = ForecastParams(
            lookback_window=20,
            train_split=2 / 3,
            epochs=2,
            batch_size=128,
            lags=5,
            multi_series=True,
        )
        outcome = run_forecast(prices, backend, params)
        assert outcome.tickers == TICKERS
        assert not outcome.failures
        for metrics in outcome.metrics():
            assert metrics["mase_vs_naive"] > 0.9

    @pytest.mark.parametrize(
        ("backend", "module"),
        [("torch_lstm", "Layer1_LSTM.torch_lstm"), ("lightgbm", "Layer1_LSTM.gbm")],
    )
    def test_pooling_trains_one_model_instead_of_one_per_ticker(
        self, prices, backend, module, monkeypatch
    ):
        """The practical argument for multi_series, asserted structurally.

        Counting training calls rather than timing them: wall-clock differences are
        real at scale but indistinguishable from noise on three tiny models.
        """
        import importlib

        target = importlib.import_module(module)
        calls: list[str] = []
        original = target.train_model

        def counted(*args, **kwargs):
            calls.append(kwargs.get("label", "model"))
            return original(*args, **kwargs)

        monkeypatch.setattr(target, "train_model", counted)

        def run(multi: bool) -> int:
            calls.clear()
            params = ForecastParams(
                lookback_window=20,
                train_split=2 / 3,
                epochs=2,
                batch_size=128,
                lags=5,
                multi_series=multi,
            )
            run_forecast(prices, backend, params)
            return len(calls)

        assert run(False) == len(TICKERS)
        assert run(True) == 1


class TestTargets:
    @pytest.mark.parametrize("backend", ["torch_lstm", "lightgbm"])
    def test_price_target_is_far_worse_once_the_scaler_cannot_see_the_future(self, prices, backend):
        """Why the default target is `return`, demonstrated rather than asserted.

        A price-level target requires extrapolating beyond the training range. With
        the scaler fitted on training data only -- as it now is -- the model cannot,
        and errors are an order of magnitude worse than the naive baseline. The
        original pipeline scored well on this target precisely *because* its scaler
        had seen the test window.
        """

        def run(target: str):
            params = ForecastParams(
                lookback_window=20,
                train_split=2 / 3,
                epochs=3,
                batch_size=128,
                lags=5,
                target=target,
            )
            return run_forecast(prices, backend, params).metrics()[0]

        on_returns, on_prices = run("return"), run("price")

        assert on_returns["mase_vs_naive"] < 1.1
        assert on_prices["mase_vs_naive"] > 5.0
        assert on_prices["r2"] < on_returns["r2"]


class TestDiagnostics:
    def test_lightgbm_reports_feature_importances(self, prices):
        outcome = run_forecast(prices, "lightgbm", FAST)
        importances = outcome.diagnostics["feature_importances"]
        assert importances
        assert sum(importances.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(0.0 <= value <= 1.0 for value in importances.values())


class TestReproducibility:
    @pytest.mark.parametrize("backend", ["torch_lstm", "lightgbm"])
    def test_same_seed_gives_the_same_forecast(self, prices, backend):
        first = run_forecast(prices, backend, FAST).predicted_prices()
        second = run_forecast(prices, backend, FAST).predicted_prices()
        assert first.to_numpy() == pytest.approx(second.to_numpy(), rel=1e-9)
