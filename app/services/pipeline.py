"""The end-to-end run: prices -> forecast -> optimize -> construct -> score."""

from __future__ import annotations

from collections.abc import Callable

from app.schemas.common import Frame, TickerError
from app.schemas.forecast import TickerForecastMetrics
from app.schemas.pipeline import PipelineRequest, PipelineResult
from app.schemas.portfolio import StrategyPerformance
from app.services import forecasting, market_data, optimization, performance
from app.settings import get_settings

Progress = Callable[[float, str], None]


def _noop(_fraction: float, _message: str) -> None:
    return None


def run(request: PipelineRequest, report: Progress | None = None) -> PipelineResult:
    """Execute the full pipeline, reporting coarse progress as it goes."""
    report = report or _noop
    settings = get_settings()

    report(0.02, "downloading prices")
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

    def forecast_progress(fraction: float, message: str) -> None:
        # The forecast dominates the runtime, so it owns most of the bar.
        report(0.05 + 0.75 * fraction, message)

    outcome = forecasting.run_forecast(
        price_result.prices, request.backend.value, params, forecast_progress
    )

    failures = [TickerError(ticker=t, reason=r) for t, r in outcome.failures]
    failures.extend(
        TickerError(ticker=t, reason="no market data returned") for t in price_result.missing
    )

    report(0.82, "optimizing portfolios")
    predicted_returns = outcome.predicted_returns()
    strategies = [s.value for s in request.strategies] if request.strategies else None
    weights, warnings = optimization.optimize(
        predicted_returns, strategies, request.risk_free_rate
    )

    report(0.92, "scoring portfolios")
    # Portfolios are scored on *realised* returns held at the forecast-derived
    # weights. Scoring on predicted returns instead measures the forecast, not
    # the strategy, and is what made the original performance table circular.
    actual_returns = outcome.actual_returns()
    perf, _portfolio_returns, cumulative = performance.evaluate(
        actual_returns, weights, request.risk_free_rate
    )

    selected = performance.select(perf, request.risk_tolerance)
    selected_weights = weights[selected]
    total = float(selected_weights.sum())
    normalised = selected_weights / total if total else selected_weights

    report(1.0, "complete")
    return PipelineResult(
        tickers=outcome.tickers,
        failed=failures,
        backend=outcome.backend,
        forecast_metrics=[TickerForecastMetrics(**m) for m in outcome.metrics()],
        weights=Frame.from_pandas(weights.rename_axis(None)),
        performance=_performance_rows(perf),
        cumulative_growth=Frame.from_pandas(cumulative),
        selected_strategy=selected,
        selected_weights={str(k): float(v) for k, v in normalised.items()},
        warnings=warnings,
        diagnostics=outcome.diagnostics,
    )


def _performance_rows(perf) -> list[StrategyPerformance]:
    return [
        StrategyPerformance(
            strategy=str(name),
            annual_return=float(row["Annual Return"]),
            annual_volatility=float(row["Annual Vol"]),
            sharpe=float(row["Sharpe"]),
            sortino=float(row["Sortino"]),
            max_drawdown=float(row["Max Drawdown"]),
        )
        for name, row in perf.iterrows()
    ]
