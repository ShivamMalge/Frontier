"""Task functions executed by job workers.

Everything here must be importable *by the worker process* from its dotted path,
and must accept and return plain JSON-compatible types -- both sides of the broker
need to serialise them. Keep the bodies thin: they translate payloads and delegate
to a service.

Progress is reported through :func:`app.jobs.progress.report`, which resolves to
the in-process record or to the RQ job's metadata depending on who is running.
"""

from __future__ import annotations

from typing import Any

from app.jobs.progress import report
from app.schemas.backtest import BacktestRequest
from app.schemas.pipeline import PipelineRequest
from app.services import backtest as backtest_service
from app.services import pipeline as pipeline_service

#: Dotted paths. Routers reference these rather than retyping the strings, so a
#: rename cannot silently break enqueueing.
RUN_PIPELINE = "app.tasks.run_pipeline"
RUN_BACKTEST = "app.tasks.run_backtest"


def run_pipeline(request: dict[str, Any]) -> dict[str, Any]:
    """Run the full pipeline for a serialised :class:`PipelineRequest`."""
    parsed = PipelineRequest.model_validate(request)
    result = pipeline_service.run(parsed, report)
    return result.model_dump(mode="json")


def run_backtest(request: dict[str, Any]) -> dict[str, Any]:
    """Run a walk-forward backtest for a serialised :class:`BacktestRequest`."""
    from app.schemas.backtest import BacktestResponse, BacktestStrategyResult
    from app.schemas.common import Frame
    from app.services import market_data

    parsed = BacktestRequest.model_validate(request)

    report(0.02, "downloading prices")
    returns, _price_result = market_data.get_returns(parsed.tickers, parsed.start, parsed.end)

    strategies = [s.value for s in parsed.strategies] if parsed.strategies else None
    result, summary = backtest_service.run(
        returns,
        strategies=strategies,
        risk_free_rate=parsed.risk_free_rate,
        constraints=parsed.constraints.to_domain(),
        lookback=parsed.lookback,
        rebalance_every=parsed.rebalance_every,
        cost_bps=parsed.cost_bps,
        on_progress=lambda fraction, message: report(0.05 + 0.9 * fraction, message),
    )

    report(0.98, "summarising")
    response = BacktestResponse(
        # Asset tickers, not the strategy names that label the result columns.
        tickers=[str(column) for column in returns.columns],
        observations=len(result.net_returns),
        lookback=parsed.lookback,
        rebalance_every=parsed.rebalance_every,
        cost_bps=parsed.cost_bps,
        rebalance_dates=[d.date().isoformat() for d in result.rebalance_dates],
        results=[
            BacktestStrategyResult(
                strategy=str(name),
                annual_return=float(row["Annual Return"]),
                annual_volatility=float(row["Annual Vol"]),
                sharpe=float(row["Sharpe"]),
                sortino=float(row["Sortino"]),
                max_drawdown=float(row["Max Drawdown"]),
                gross_annual_return=float(row["Gross Annual Return"]),
                cost_drag=float(row["Cost Drag"]),
                avg_turnover=float(row["Avg Turnover"]),
                annual_turnover=float(row["Annual Turnover"]),
                total_cost=float(row["Total Cost"]),
                rebalances=int(row["Rebalances"]),
            )
            for name, row in summary.iterrows()
        ],
        cumulative_growth=Frame.from_pandas(result.cumulative(net=True)),
        gross_cumulative_growth=Frame.from_pandas(result.cumulative(net=False)),
        turnover=Frame.from_pandas(result.turnover),
        warnings=result.warnings,
    )
    report(1.0, "complete")
    return response.model_dump(mode="json")
