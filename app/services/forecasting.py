"""Forecasting with pluggable backends.

A backend is registered with a *scope*:

``ticker``
    Called once per series. Simple, and what the legacy models need.
``universe``
    Called once with the whole price frame, so it can train a single model across
    every ticker. This is what makes ``multi_series`` possible: one training run
    instead of eighteen, and a model that can learn structure repeating across
    names.

Availability is checked at import time per backend, so a deployment without
PyTorch or TensorFlow advertises an honest capability set through ``/health``
rather than failing on the first request.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from app.errors import InsufficientDataError, UnknownBackendError
from Layer1_LSTM.results import SeriesForecast
from utils.config import MIN_OBSERVATIONS

Scope = Literal["ticker", "universe"]


@dataclass(frozen=True)
class ForecastParams:
    lookback_window: int
    train_split: float
    epochs: int
    batch_size: int
    target: str = "return"
    multi_series: bool = False
    lags: int = 10
    seed: int = 42


@dataclass(frozen=True)
class BackendSpec:
    name: str
    scope: Scope
    #: Import name that must be present for this backend to be usable.
    requires: str | None
    description: str
    fn: Callable[..., object]

    @property
    def available(self) -> bool:
        return self.requires is None or importlib.util.find_spec(self.requires) is not None


_REGISTRY: dict[str, BackendSpec] = {}


def register(
    name: str, scope: Scope, description: str, requires: str | None = None
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    def decorator(fn: Callable[..., object]) -> Callable[..., object]:
        _REGISTRY[name] = BackendSpec(name, scope, requires, description, fn)
        return fn

    return decorator


def available_backends() -> list[str]:
    return sorted(name for name, spec in _REGISTRY.items() if spec.available)


def backend_specs() -> list[BackendSpec]:
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


# --------------------------------------------------------------------------
# ticker-scope backends
# --------------------------------------------------------------------------


@register(
    "naive",
    scope="ticker",
    description="Random walk: tomorrow equals today. The baseline every other "
    "backend is measured against; scores MASE 1.0 by definition.",
)
def _naive(series: pd.Series, params: ForecastParams) -> SeriesForecast:
    import numpy as np

    values = series.to_numpy(dtype=float)
    if values.size < MIN_OBSERVATIONS:
        raise InsufficientDataError(
            f"needs at least {MIN_OBSERVATIONS} observations, got {values.size}",
            ticker=str(series.name),
        )

    start = int(np.ceil(values.size * params.train_split))
    return SeriesForecast(
        ticker=str(series.name),
        dates=series.index[start:],
        predicted=values[start - 1 : -1],
        actual=values[start:],
    )


@register(
    "keras_lstm",
    scope="ticker",
    requires="tensorflow",
    description="Legacy TensorFlow/Keras LSTM: one model per ticker, univariate "
    "raw prices, price-level target. Superseded by torch_lstm; kept so the "
    "original result can be reproduced. Scaler leakage is now fixed.",
)
def _keras_lstm(series: pd.Series, params: ForecastParams) -> SeriesForecast:
    import numpy as np

    from Layer1_LSTM.evaluate_lstm import lstm_predict
    from Layer1_LSTM.train_lstm import train_lstm_model

    values = series.to_numpy(dtype=float).reshape(-1, 1)
    if values.shape[0] < MIN_OBSERVATIONS:
        raise InsufficientDataError(
            f"needs at least {MIN_OBSERVATIONS} observations, got {values.shape[0]}",
            ticker=str(series.name),
        )

    model, scaler, train_len = train_lstm_model(
        values,
        lookback_window=params.lookback_window,
        train_split=params.train_split,
        epochs=params.epochs,
        batch_size=params.batch_size,
    )
    predicted, actual = lstm_predict(model, scaler, values, train_len, params.lookback_window)
    if predicted is None:
        raise InsufficientDataError(
            "not enough test observations to form a single lookback window",
            ticker=str(series.name),
        )

    return SeriesForecast(
        ticker=str(series.name),
        dates=series.index[-predicted.size :],
        predicted=np.asarray(predicted, dtype=float),
        actual=np.asarray(actual, dtype=float),
    )


# --------------------------------------------------------------------------
# universe-scope backends
# --------------------------------------------------------------------------


@register(
    "torch_lstm",
    scope="universe",
    requires="torch",
    description="PyTorch LSTM on 22 engineered features, return target, "
    "leak-free scaling and early stopping on a chronological validation tail. "
    "Set multi_series to train one model across the whole universe.",
)
def _torch_lstm(prices: pd.DataFrame, params: ForecastParams, on_progress=None):
    from Layer1_LSTM.torch_lstm import TorchConfig, forecast_universe

    config = TorchConfig(
        window=params.lookback_window,
        epochs=params.epochs,
        batch_size=params.batch_size,
        target=params.target,  # type: ignore[arg-type]
        multi_series=params.multi_series,
        lags=params.lags,
        train_split=params.train_split,
        seed=params.seed,
    )
    return forecast_universe(prices, config, on_progress)


@register(
    "lightgbm",
    scope="universe",
    requires="lightgbm",
    description="Gradient-boosted trees on the same engineered features. Trains "
    "the whole universe in seconds and reports feature importances. Usually the "
    "stronger baseline on daily equity data.",
)
def _lightgbm(prices: pd.DataFrame, params: ForecastParams, on_progress=None):
    from Layer1_LSTM.gbm import GBMConfig, forecast_universe

    config = GBMConfig(
        target=params.target,  # type: ignore[arg-type]
        multi_series=params.multi_series,
        lags=params.lags,
        train_split=params.train_split,
        seed=params.seed,
    )
    forecasts, failures = forecast_universe(prices, config, on_progress)
    return forecasts, failures, {"feature_importances": config.feature_importances}


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


@dataclass
class ForecastOutcome:
    backend: str
    params: ForecastParams
    forecasts: list[SeriesForecast]
    failures: list[tuple[str, str]]
    #: Backend-specific extras, e.g. LightGBM feature importances.
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def tickers(self) -> list[str]:
        return [forecast.ticker for forecast in self.forecasts]

    def _frame(self, attribute: str) -> pd.DataFrame:
        if not self.forecasts:
            return pd.DataFrame()
        # Aligned on dates, so tickers with differing histories line up by date
        # rather than by position.
        return pd.DataFrame(
            {
                forecast.ticker: pd.Series(getattr(forecast, attribute), index=forecast.dates)
                for forecast in self.forecasts
            }
        ).sort_index()

    def predicted_prices(self) -> pd.DataFrame:
        return self._frame("predicted")

    def actual_prices(self) -> pd.DataFrame:
        return self._frame("actual")

    def predicted_returns(self) -> pd.DataFrame:
        """Returns implied by the forecast -- what the optimizers consume."""
        from Layer1_Preprocessing.preprocessing import compute_returns

        return compute_returns(self.predicted_prices())

    def actual_returns(self) -> pd.DataFrame:
        from Layer1_Preprocessing.preprocessing import compute_returns

        return compute_returns(self.actual_prices())

    def metrics(self) -> list[dict[str, float | str]]:
        return [{"ticker": f.ticker, **f.metrics()} for f in self.forecasts]


def run_forecast(
    prices: pd.DataFrame,
    backend: str,
    params: ForecastParams,
    on_progress: Callable[[float, str], None] | None = None,
) -> ForecastOutcome:
    """Forecast every column of ``prices`` using ``backend``."""
    spec = _REGISTRY.get(backend)
    if spec is None or not spec.available:
        raise UnknownBackendError(
            f"forecast backend '{backend}' is not available in this process",
            available=",".join(available_backends()),
        )

    if spec.scope == "universe":
        result = spec.fn(prices, params, on_progress)
        forecasts, failures, diagnostics = _unpack(result)
        if on_progress:
            on_progress(1.0, "forecast complete")
        return ForecastOutcome(backend, params, forecasts, failures, diagnostics)

    forecasts, failures = [], []
    total = max(len(prices.columns), 1)

    for position, ticker in enumerate(prices.columns):
        if on_progress:
            on_progress(position / total, f"forecasting {ticker}")
        series = prices[ticker].dropna()
        series.name = ticker
        try:
            forecasts.append(spec.fn(series, params))
        except Exception as exc:  # noqa: BLE001 -- one bad ticker must not sink the request
            failures.append((str(ticker), f"{type(exc).__name__}: {exc}"))

    if on_progress:
        on_progress(1.0, "forecast complete")
    return ForecastOutcome(backend, params, forecasts, failures)


def _unpack(result) -> tuple[list[SeriesForecast], list[tuple[str, str]], dict[str, object]]:
    """Universe backends may return two or three values."""
    if len(result) == 3:
        forecasts, failures, diagnostics = result
        return list(forecasts), list(failures), dict(diagnostics)
    forecasts, failures = result
    return list(forecasts), list(failures), {}
