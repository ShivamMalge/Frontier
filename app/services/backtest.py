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
from app.services import optimization
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
            except Exception as exc:
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
) -> tuple[BacktestResult, pd.DataFrame]:
    """Run the backtest and return ``(result, summary)``."""
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
