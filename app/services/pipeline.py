"""The end-to-end run: prices -> forecast -> optimize -> construct -> score."""

from __future__ import annotations

from collections.abc import Callable

from app.schemas.common import Frame, TickerError
from app.schemas.forecast import TickerForecastMetrics
from app.schemas.pipeline import PipelineRequest, PipelineResult
from app.schemas.portfolio import StrategyPerformance
from app.services import forecasting, market_data, optimization, performance, tracking
from app.settings import get_settings

Progress = Callable[[float, str], None]


def _noop(_fraction: float, _message: str) -> None:
    return None


def run(request: PipelineRequest, report: Progress | None = None) -> PipelineResult:
    """Execute the full pipeline, reporting coarse progress as it goes.

    The whole run is tracked as one MLflow experiment when tracking is enabled, so
    the forecast metrics and the strategy performance that followed from them stay
    attached to each other and to the data they came from.
    """
    with tracking.track(
        f"pipeline:{request.backend.value}",
        tags={"kind": "pipeline", **tracking.data_vintage_tags(request.tickers)},
    ) as run_handle:
        return _run(request, report or _noop, run_handle)


def _run(request: PipelineRequest, report: Progress, run_handle: tracking.Run) -> PipelineResult:
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

    run_handle.log_params(
        {
            "tickers": request.tickers,
            "n_tickers": len(request.tickers),
            "backend": request.backend.value,
            "target": request.target.value,
            "multi_series": request.multi_series,
            "lookback_window": params.lookback_window,
            "train_split": params.train_split,
            "epochs": params.epochs,
            "batch_size": params.batch_size,
            "lags": params.lags,
            "seed": params.seed,
            "start": request.start or settings.start_date,
            "end": request.end or settings.end_date,
            "risk_free_rate": (
                settings.risk_free_rate
                if request.risk_free_rate is None
                else request.risk_free_rate
            ),
            "risk_tolerance": request.risk_tolerance,
            "observations": len(price_result.prices),
        }
    )

    outcome = forecasting.run_forecast(
        price_result.prices, request.backend.value, params, forecast_progress
    )

    failures = [TickerError(ticker=t, reason=r) for t, r in outcome.failures]
    failures.extend(
        TickerError(ticker=t, reason="no market data returned") for t in price_result.missing
    )

    # Forecast quality, per ticker and aggregated. Logging the baseline comparison
    # (mase_vs_naive) next to the legacy figure is the point: the original project
    # reported 93% with no record of what it measured.
    forecast_metrics = outcome.metrics()
    run_handle.log_table(forecast_metrics, "forecast_metrics.json")
    for name in (
        "mase_vs_naive",
        "directional_accuracy",
        "r2",
        "rmse",
        "legacy_approximate_accuracy",
    ):
        run_handle.log_metrics(
            tracking.aggregate([row.get(name) for row in forecast_metrics], name)
        )

    report(0.82, "optimizing portfolios")
    predicted_returns = outcome.predicted_returns()
    strategies = [s.value for s in request.strategies] if request.strategies else None
    weights, warnings = optimization.optimize(
        predicted_returns,
        strategies,
        request.risk_free_rate,
        request.constraints.to_domain(),
    )

    report(0.92, "scoring portfolios")
    # Portfolios are scored on *realised* returns held at the forecast-derived
    # weights. Scoring on predicted returns instead measures the forecast, not
    # the strategy, and is what made the original performance table circular.
    actual_returns = outcome.actual_returns()
    perf, _portfolio_returns, cumulative = performance.evaluate(
        actual_returns, weights, request.risk_free_rate
    )

    performance_rows = _performance_rows(perf)
    run_handle.log_table(
        [row.model_dump() for row in performance_rows], "strategy_performance.json"
    )
    for row in performance_rows:
        run_handle.log_metrics(
            {
                "annual_return": row.annual_return,
                "annual_volatility": row.annual_volatility,
                "sharpe": row.sharpe,
                "sortino": row.sortino,
                "max_drawdown": row.max_drawdown,
            },
            prefix=f"{row.strategy}__",
        )

    selected = performance.select(perf, request.risk_tolerance)
    selected_weights = weights[selected]
    total = float(selected_weights.sum())
    normalised = selected_weights / total if total else selected_weights

    run_handle.log_tags(
        {
            "selected_strategy": selected,
            "tickers_forecast": len(outcome.tickers),
            "tickers_failed": len(failures),
        }
    )
    run_handle.log_metrics({"strategies": len(weights.columns)})

    report(1.0, "complete")
    return PipelineResult(
        tickers=outcome.tickers,
        failed=failures,
        backend=outcome.backend,
        forecast_metrics=[TickerForecastMetrics(**m) for m in forecast_metrics],
        weights=Frame.from_pandas(weights.rename_axis(None)),
        performance=performance_rows,
        cumulative_growth=Frame.from_pandas(cumulative),
        selected_strategy=selected,
        selected_weights={str(k): float(v) for k, v in normalised.items()},
        warnings=warnings,
        diagnostics=outcome.diagnostics,
        tracking_run_id=run_handle.run_id,
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
