"""Forecasting endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from frontier.api.schemas.common import Frame, TickerError
from frontier.api.schemas.forecast import (
    ForecastRequest,
    ForecastResponse,
    TickerForecastMetrics,
)
from frontier.services import forecasting, market_data
from frontier.settings import get_settings

router = APIRouter(tags=["forecast"])


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    summary="Forecast prices and derive predicted returns",
    description=(
        "Runs synchronously. The 'naive' backend is fast; 'keras_lstm' trains one model "
        "per ticker and can take minutes per ticker on CPU -- submit that through "
        "POST /pipeline/runs instead, which returns a job you can poll."
    ),
)
def forecast(request: ForecastRequest) -> ForecastResponse:
    settings = get_settings()
    price_result = market_data.get_prices(request.tickers, request.start, request.end)

    params = forecasting.ForecastParams(
        lookback_window=request.lookback_window or settings.lookback_window,
        train_split=request.train_split or settings.train_split,
        epochs=request.epochs or settings.lstm_epochs,
        batch_size=request.batch_size or settings.lstm_batch_size,
        target=request.target.value,
        multi_series=request.multi_series,
        lags=request.lags,
        seed=request.seed,
    )
    outcome = forecasting.run_forecast(price_result.prices, request.backend.value, params)

    failed = [TickerError(ticker=t, reason=r) for t, r in outcome.failures]
    failed.extend(
        TickerError(ticker=t, reason="no market data returned") for t in price_result.missing
    )

    return ForecastResponse(
        backend=request.backend,
        target=request.target,
        multi_series=request.multi_series,
        lookback_window=params.lookback_window,
        train_split=params.train_split,
        tickers=outcome.tickers,
        failed=failed,
        metrics=[TickerForecastMetrics(**m) for m in outcome.metrics()],
        predicted_prices=Frame.from_pandas(outcome.predicted_prices()),
        actual_prices=Frame.from_pandas(outcome.actual_prices()),
        predicted_returns=Frame.from_pandas(outcome.predicted_returns()),
        actual_returns=Frame.from_pandas(outcome.actual_returns()),
        diagnostics=outcome.diagnostics,
    )
