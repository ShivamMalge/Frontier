"""Walk-forward backtesting.

Bridges the Layer 3 engine to the Layer 2 optimizers. The engine is given a
callable rather than importing the dispatch itself, which keeps Layer 3 free of any
dependency on Layer 2 and lets the tests drive it with a trivial allocator.

One wrinkle handled here: each strategy holds a different portfolio, so each faces a
different trade to reach its target. When a turnover budget is in force that has to
be per strategy, which means one optimizer call each; without one, a single batched
call covers every strategy at once. The engine does not need to know which.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable

import pandas as pd

from app.errors import InsufficientDataError, OptimizationFailedError
from app.services import optimization, tracking
from app.settings import get_settings
from Layer2_Optimization.constraints import DEFAULT, Constraints
from Layer3_Portfolio_Generation.backtest import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
)

logger = logging.getLogger(__name__)


def make_optimizer(
    strategies: list[str] | None,
    risk_free_rate: float | None,
    constraints: Constraints,
) -> Callable[[pd.DataFrame, dict[str, dict[str, float]]], tuple[pd.DataFrame, list[str]]]:
    """Build the callable the engine invokes at each rebalance."""
    needs_per_strategy = constraints.max_turnover is not None

    def optimise(
        window: pd.DataFrame, previous: dict[str, dict[str, float]]
    ) -> tuple[pd.DataFrame, list[str]]:
        if not needs_per_strategy:
            return optimization.optimize(window, strategies, risk_free_rate, constraints)

        # A turnover budget is relative to what each strategy currently holds, so
        # each needs its own solve against its own starting point.
        selected = list(strategies) if strategies else list(optimization.ALL_STRATEGIES)
        columns: dict[str, pd.Series] = {}
        messages: list[str] = []

        for name in selected:
            holdings = previous.get(name, {})
            scoped = dataclasses.replace(
                constraints,
                # Nothing is held yet on the first rebalance, so a turnover cap
                # would make buying in impossible. Let the opening trade through.
                max_turnover=constraints.max_turnover if holdings else None,
                previous_weights=holdings,
            )
            try:
                frame, notes = optimization.optimize(window, [name], risk_free_rate, scoped)
            except Exception as exc:  # noqa: BLE001 -- one strategy, not the run
                messages.append(f"{name}: failed ({type(exc).__name__}: {exc})")
                continue
            columns[name] = frame[name]
            messages.extend(notes)

        if not columns:
            raise OptimizationFailedError("every strategy failed at this rebalance")
        return pd.DataFrame(columns), messages

    return optimise


def run(
    returns: pd.DataFrame,
    strategies: list[str] | None = None,
    risk_free_rate: float | None = None,
    constraints: Constraints = DEFAULT,
    lookback: int = 252,
    rebalance_every: int = 21,
    cost_bps: float = 10.0,
    on_progress: Callable[[float, str], None] | None = None,
) -> tuple[BacktestResult, pd.DataFrame, str | None]:
    """Run the backtest and return ``(result, summary, tracking_run_id)``.

    Tracked as its own MLflow run. A walk-forward result is the only out-of-sample
    number this project produces, so it is the one most worth being able to look up
    later alongside the parameters that produced it.
    """
    tickers = [str(column) for column in returns.columns]
    with tracking.track(
        "backtest", tags={"kind": "backtest", **tracking.data_vintage_tags(tickers)}
    ) as run_handle:
        result, summary = _run(
            returns,
            strategies,
            risk_free_rate,
            constraints,
            lookback,
            rebalance_every,
            cost_bps,
            on_progress,
        )

        run_handle.log_params(
            {
                "tickers": tickers,
                "n_tickers": len(tickers),
                "strategies": strategies or list(optimization.ALL_STRATEGIES),
                "lookback": lookback,
                "rebalance_every": rebalance_every,
                "cost_bps": cost_bps,
                "risk_free_rate": risk_free_rate,
                "min_weight": constraints.min_weight,
                "max_weight": constraints.max_weight,
                "max_leverage": constraints.max_leverage,
                "max_turnover": constraints.max_turnover,
                "observations": len(result.net_returns),
                "rebalances": len(result.rebalance_dates),
            }
        )
        run_handle.log_tags(
            {
                "first_oos_date": str(result.net_returns.index[0].date())
                if len(result.net_returns)
                else None,
                "last_oos_date": str(result.net_returns.index[-1].date())
                if len(result.net_returns)
                else None,
                "warnings": len(result.warnings),
            }
        )

        rows = [
            {"strategy": str(name), **{str(k): float(v) for k, v in row.items()}}
            for name, row in summary.iterrows()
        ]
        run_handle.log_table(rows, "backtest_summary.json")
        for row in rows:
            strategy = row.pop("strategy")
            run_handle.log_metrics(row, prefix=f"{strategy}__")

        for column in ("Sharpe", "Annual Return", "Cost Drag", "Annual Turnover"):
            if column in summary.columns:
                run_handle.log_metrics(
                    tracking.aggregate(summary[column].tolist(), column.lower().replace(" ", "_"))
                )

        return result, summary, run_handle.run_id


def _run(
    returns: pd.DataFrame,
    strategies: list[str] | None,
    risk_free_rate: float | None,
    constraints: Constraints,
    lookback: int,
    rebalance_every: int,
    cost_bps: float,
    on_progress: Callable[[float, str], None] | None,
) -> tuple[BacktestResult, pd.DataFrame]:
    settings = get_settings()

    clean = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if clean.shape[1] < 2:
        raise InsufficientDataError(f"need at least 2 assets, got {clean.shape[1]}")
    if len(clean) <= lookback:
        raise InsufficientDataError(
            f"need more than lookback={lookback} observations, got {len(clean)}; "
            "shorten the lookback or widen the date range"
        )

    config = BacktestConfig(
        lookback=lookback,
        rebalance_every=rebalance_every,
        cost_bps=cost_bps,
        periods_per_year=settings.trading_days_per_year,
    )
    optimiser = make_optimizer(strategies, risk_free_rate, constraints)

    try:
        result = run_backtest(clean, optimiser, config, on_progress)
    except ValueError as exc:
        raise InsufficientDataError(str(exc)) from exc

    rf = settings.risk_free_rate if risk_free_rate is None else risk_free_rate
    return result, result.summary(settings.trading_days_per_year, rf)
