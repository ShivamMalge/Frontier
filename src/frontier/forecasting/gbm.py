# src/frontier/forecasting/gbm.py

"""LightGBM forecaster.

Gradient-boosted trees on the engineered features from
``frontier.data.feature_engineering``. Worth having alongside the LSTM for
reasons that are practical rather than fashionable:

* **seconds, not minutes.** Training the whole universe is roughly three orders of
  magnitude cheaper than eighteen LSTMs, which is what made the original pipeline
  unusable interactively.
* **it reads tabular features natively**, so volatility, momentum and calendar
  effects go in directly without being flattened into a sequence.
* **feature importances** say which inputs the model actually used -- something an
  LSTM does not offer, and useful evidence when writing this up.
* **it is the stronger baseline.** On daily equity data, boosted trees on lagged
  features typically match or beat a univariate LSTM. If the LSTM cannot beat this,
  that is the finding.

Like the LSTM it early-stops on a chronological tail of the training window, never
a random split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .datasets import SplitDataset, Target, build_splits
from .results import SeriesForecast

logger = logging.getLogger(__name__)


@dataclass
class GBMConfig:
    n_estimators: int = 400
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 40
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    #: L1/L2 regularisation. Financial features are noisy and highly correlated;
    #: unregularised trees overfit them readily.
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    validation_fraction: float = 0.15
    early_stopping_rounds: int = 40
    seed: int = 42
    target: Target = "return"
    multi_series: bool = False
    lags: int = 10
    train_split: float = 2 / 3
    n_jobs: int = -1
    feature_importances: dict[str, float] = field(default_factory=dict)
    best_iteration: int | None = None


def _regressor(config: GBMConfig):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="regression",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        subsample_freq=config.subsample_freq,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=config.seed,
        n_jobs=config.n_jobs,
        verbose=-1,
    )


def train_model(x: np.ndarray, y: np.ndarray, config: GBMConfig, label: str = "model"):
    """Fit one booster with early stopping on the chronological tail."""
    from lightgbm import early_stopping, log_evaluation

    cut = int(len(y) * (1.0 - config.validation_fraction))
    cut = max(1, min(cut, len(y) - 1)) if len(y) > 1 else len(y)

    model = _regressor(config)
    fit_kwargs = {}
    if cut < len(y):
        fit_kwargs = {
            "eval_X": x[cut:],
            "eval_y": y[cut:],
            "eval_metric": "l2",
            "callbacks": [
                early_stopping(config.early_stopping_rounds, verbose=False),
                log_evaluation(period=0),
            ],
        }

    model.fit(x[:cut], y[:cut], **fit_kwargs)
    logger.info("%s: best iteration %s of %s", label, model.best_iteration_, config.n_estimators)
    return model


def predict(model, x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(0)
    return np.asarray(model.predict(x), dtype=float).ravel()


def forecast_universe(
    prices: pd.DataFrame,
    config: GBMConfig,
    on_progress=None,
) -> tuple[list[SeriesForecast], list[tuple[str, str]]]:
    """Forecast every column of ``prices``, one booster or one shared booster."""
    # One Polars pass builds features for every ticker, rather than looping in
    # Python and rebuilding the same 22 columns per series.
    splits, failures = build_splits(
        prices,
        train_split=config.train_split,
        target=config.target,
        lags=config.lags,
        window=None,
    )

    if not splits:
        return [], failures

    forecasts: list[SeriesForecast] = []
    first = next(iter(splits.values()))

    if config.multi_series:
        if on_progress:
            on_progress(0.2, f"training one booster across {len(splits)} tickers")
        x = np.concatenate([split.x_train for split in splits.values()])
        y = np.concatenate([split.y_train for split in splits.values()])
        model = train_model(x, y, config, label="multi-series")
        _record_diagnostics(model, first, config)

        for position, (ticker, split) in enumerate(splits.items(), start=1):
            if on_progress:
                on_progress(0.6 + 0.4 * position / len(splits), f"predicting {ticker}")
            forecasts.append(_to_forecast(model, split, config))
        return forecasts, failures

    for position, (ticker, split) in enumerate(splits.items(), start=1):
        if on_progress:
            on_progress(position / len(splits), f"training {ticker}")
        try:
            model = train_model(split.x_train, split.y_train, config, label=ticker)
            _record_diagnostics(model, split, config)
            forecasts.append(_to_forecast(model, split, config))
        except Exception as exc:  # noqa: BLE001 -- a ticker that will not train is reported, not fatal
            failures.append((ticker, f"{type(exc).__name__}: {exc}"))

    return forecasts, failures


def _record_diagnostics(model, split: SplitDataset, config: GBMConfig) -> None:
    """Keep the last model's importances; surfaced by the API for inspection."""
    importances = np.asarray(model.feature_importances_, dtype=float)
    total = importances.sum() or 1.0
    config.feature_importances = {
        name: float(value / total)
        for name, value in zip(split.feature_names, importances, strict=False)
    }
    config.best_iteration = model.best_iteration_


def _to_forecast(model, split: SplitDataset, config: GBMConfig) -> SeriesForecast:
    return SeriesForecast(
        ticker=split.ticker,
        dates=split.test_dates,
        predicted=split.rebuild_prices(predict(model, split.x_test), config.target),
        actual=split.test_actual_price,
    )
