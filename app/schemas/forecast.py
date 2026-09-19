"""Forecasting contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import Frame, TickerError, WindowRequest


class ForecastBackend(StrEnum):
    """Pluggable forecasting implementations.

    ``NAIVE`` is the random-walk baseline (tomorrow == today). It exists because a
    price forecaster can only be judged relative to doing nothing, and it scores
    MASE 1.0 by definition.

    ``TORCH_LSTM`` and ``LIGHTGBM`` are the Phase 3 models: multivariate features,
    a return target, leak-free scaling and early stopping. ``KERAS_LSTM`` is the
    original univariate price-level model, kept so the two can be compared.
    """

    NAIVE = "naive"
    KERAS_LSTM = "keras_lstm"
    TORCH_LSTM = "torch_lstm"
    LIGHTGBM = "lightgbm"


class ForecastTarget(StrEnum):
    """What the model predicts.

    ``RETURN`` -- tomorrow's simple return, reconstructed into a price level
    against the last observed close. This is the actual forecasting problem.

    ``PRICE`` -- tomorrow's price level, as the original pipeline did. Predicting a
    price from a window of prices is close to an identity mapping, which is why it
    scores well on R^2 and MAPE while carrying almost no information. Available so
    the difference can be demonstrated rather than asserted.
    """

    RETURN = "return"
    PRICE = "price"


class ForecastRequest(WindowRequest):
    backend: ForecastBackend = Field(
        default=ForecastBackend.NAIVE,
        description="Forecasting implementation. 'lightgbm' trains the whole universe "
        "in seconds; 'torch_lstm' and 'keras_lstm' take minutes and are better "
        "submitted through POST /pipeline/runs.",
    )
    target: ForecastTarget = Field(
        default=ForecastTarget.RETURN,
        description="Ignored by 'naive' and 'keras_lstm', which always model price levels.",
    )
    multi_series: bool = Field(
        default=False,
        description="Train one model across the whole universe instead of one per "
        "ticker. Supported by 'torch_lstm' and 'lightgbm'; far cheaper, and lets the "
        "model use structure shared between tickers.",
    )
    lookback_window: int | None = Field(default=None, ge=2, le=512)
    train_split: float | None = Field(default=None, gt=0.1, lt=0.95)
    epochs: int | None = Field(default=None, ge=1, le=500)
    batch_size: int | None = Field(default=None, ge=1, le=4096)
    lags: int = Field(
        default=10, ge=1, le=60, description="Number of lagged returns among the features."
    )
    seed: int = Field(default=42, description="Seed for reproducible training.")


class TickerForecastMetrics(BaseModel):
    """Out-of-sample forecast quality for one ticker.

    ``rmse``/``mae``/``mape``/``r2`` are computed on price levels and are therefore
    scale-dependent and easy to misread: R^2 is near 1.0 for any trending series,
    and MAPE is small for any forecast close to the last observed price.
    ``directional_accuracy`` and ``mase_vs_naive`` are the two fields that
    actually carry signal.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    observations: int
    rmse: float
    mae: float
    r2: float
    mape: float
    directional_accuracy: float | None = Field(
        description="Share of the model's directional calls that were correct; 0.5 is a "
        "coin flip. Null when the model made no directional call at all, which is the "
        "case for the flat random-walk baseline."
    )
    mase_vs_naive: float = Field(
        description="Mean absolute error divided by that of a random-walk forecast. "
        "Below 1.0 beats the naive baseline; at or above 1.0 it does not."
    )
    legacy_approximate_accuracy: float = Field(
        description="Retained for continuity with earlier reports: 100 * (1 - RMSE / mean price). "
        "Not a measure of forecast skill -- a random walk typically scores above 97 on it."
    )


class ForecastResponse(BaseModel):
    backend: ForecastBackend
    target: ForecastTarget
    multi_series: bool
    lookback_window: int
    train_split: float
    tickers: list[str] = Field(description="Tickers successfully forecast.")
    failed: list[TickerError] = Field(default_factory=list)
    metrics: list[TickerForecastMetrics]
    predicted_prices: Frame = Field(description="Model output, in price units.")
    actual_prices: Frame = Field(description="Realised prices over the same window.")
    predicted_returns: Frame = Field(
        description="Simple returns derived from predicted_prices. This -- not the price "
        "level -- is what the optimizers consume."
    )
    actual_returns: Frame
    diagnostics: dict[str, object] = Field(
        default_factory=dict,
        description="Backend extras, e.g. LightGBM feature importances as fractions of "
        "total split gain.",
    )
