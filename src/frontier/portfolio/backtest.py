# src/frontier/portfolio/backtest.py

"""Walk-forward backtest with transaction costs.

Closes the last substantive gap in the evaluation. Until now weights were fitted
on the same window used to score them, so the performance table said "how well
would this have done if we had known the answer" -- flattering and uninformative.
Here every allocation is estimated from a *trailing* window and then held forward
through data the optimizer never saw.

Three details that decide whether a backtest means anything:

**Weights drift.** Between rebalances the portfolio is held, not continuously
rebalanced, so weights move with prices: ``w_i' = w_i (1 + r_i) / (1 + R)``.
Pinning weights to their targets every day implies free daily trading and quietly
inflates returns.

**Turnover is measured against the drifted weights**, not against the previous
target. The difference is what actually has to be traded.

**Costs are charged.** ``cost_bps`` is applied to traded notional at each
rebalance, so a strategy that churns pays for it. A backtest without costs makes
high-turnover strategies look better than they are.

The estimation window never includes the day being predicted, and the optimizer is
handed the drifted weights so a ``max_turnover`` constraint can act on reality.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Signature of the optimizer the backtest calls at each rebalance.
#:
#: Receives the trailing return window and, keyed by strategy, that strategy's
#: current drifted weights -- each strategy holds a different portfolio, so each
#: faces a different trade to reach its target and a different turnover budget.
#: Returns a tickers x strategies frame plus any warnings. Injected rather than
#: imported so Layer 3 stays independent of Layer 2's dispatch, and so tests can
#: drive the engine with a trivial allocator.
Optimizer = Callable[[pd.DataFrame, dict[str, dict[str, float]]], "tuple[pd.DataFrame, list[str]]"]


@dataclass(frozen=True)
class BacktestConfig:
    #: Trailing observations used to estimate expected returns and covariance.
    lookback: int = 252
    #: Trading days between rebalances. 21 is roughly monthly.
    rebalance_every: int = 21
    #: Cost in basis points per unit of traded notional, charged on ``sum |dw|``
    #: at each rebalance. 10 bps on a 50% turnover costs 5 bps of portfolio value.
    cost_bps: float = 10.0
    #: Annualisation factor for the summary statistics.
    periods_per_year: int = 252


#: Monthly rebalancing on a one-year window at 10 bps. Frozen, so sharing one
#: instance as a default is safe.
DEFAULT_CONFIG = BacktestConfig()


@dataclass
class BacktestResult:
    """Out-of-sample results, one column per strategy."""

    net_returns: pd.DataFrame
    gross_returns: pd.DataFrame
    #: Weights targeted at each rebalance, keyed by strategy.
    weights: dict[str, pd.DataFrame] = field(default_factory=dict)
    #: ``sum |dw|`` at each rebalance, one column per strategy.
    turnover: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Cost charged at each rebalance, as a fraction of portfolio value.
    costs: pd.DataFrame = field(default_factory=pd.DataFrame)
    rebalance_dates: list[pd.Timestamp] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def cumulative(self, net: bool = True) -> pd.DataFrame:
        source = self.net_returns if net else self.gross_returns
        return (1.0 + source).cumprod()

    def summary(self, periods_per_year: int = 252, rf: float = 0.0) -> pd.DataFrame:
        """Annualised statistics, gross and net, plus turnover and total cost."""
        from .compare import compute_performance

        net = compute_performance(self.net_returns, rf=rf)
        gross = compute_performance(self.gross_returns, rf=rf)

        rows = {}
        for strategy in self.net_returns.columns:
            turnover = float(self.turnover[strategy].mean()) if strategy in self.turnover else 0.0
            total_cost = float(self.costs[strategy].sum()) if strategy in self.costs else 0.0
            rows[strategy] = {
                "Annual Return": net.loc[strategy, "Annual Return"],
                "Annual Vol": net.loc[strategy, "Annual Vol"],
                "Sharpe": net.loc[strategy, "Sharpe"],
                "Sortino": net.loc[strategy, "Sortino"],
                "Max Drawdown": net.loc[strategy, "Max Drawdown"],
                "Gross Annual Return": gross.loc[strategy, "Annual Return"],
                "Cost Drag": gross.loc[strategy, "Annual Return"]
                - net.loc[strategy, "Annual Return"],
                "Avg Turnover": turnover,
                "Annual Turnover": turnover * (periods_per_year / max(1, _spacing(self))),
                "Total Cost": total_cost,
                "Rebalances": len(self.rebalance_dates),
            }
        return pd.DataFrame(rows).T


def _spacing(result: BacktestResult) -> int:
    if len(result.rebalance_dates) < 2:
        return 1
    return max(1, round(len(result.net_returns) / max(1, len(result.rebalance_dates))))


def rebalance_schedule(n_observations: int, lookback: int, every: int) -> list[int]:
    """Indices at which to re-optimise.

    The first is ``lookback``, so the opening allocation already has a full
    estimation window behind it and nothing is fitted on the future.
    """
    if n_observations <= lookback:
        return []
    return list(range(lookback, n_observations, max(1, every)))


def run_backtest(  # noqa: PLR0912, PLR0915 -- one sequential walk; splitting it scatters the state
    returns: pd.DataFrame,
    optimizer: Optimizer,
    config: BacktestConfig = DEFAULT_CONFIG,
    on_progress: Callable[[float, str], None] | None = None,
) -> BacktestResult:
    """Walk ``returns`` forward, re-optimising on a trailing window.

    ``returns`` are realised asset returns. At each rebalance index ``t`` the
    optimizer sees only ``returns.iloc[t - lookback : t]``; day ``t`` onwards is
    then held with the weights that window produced.
    """
    clean = returns.dropna(axis=1, how="all").dropna(axis=0, how="any").sort_index()
    assets = [str(column) for column in clean.columns]
    schedule = rebalance_schedule(len(clean), config.lookback, config.rebalance_every)

    if not schedule:
        raise ValueError(
            f"need more than lookback={config.lookback} observations to backtest, got {len(clean)}"
        )
    if len(assets) < 2:
        raise ValueError(f"need at least 2 assets to backtest, got {len(assets)}")

    values = clean.to_numpy(dtype=float)
    dates = clean.index
    cost_rate = config.cost_bps / 10_000.0
    boundaries = [*schedule, len(clean)]

    strategies: list[str] | None = None
    held: dict[str, np.ndarray] = {}
    gross: dict[str, list[float]] = {}
    net: dict[str, list[float]] = {}
    targets: dict[str, dict[pd.Timestamp, np.ndarray]] = {}
    turnovers: dict[str, dict[pd.Timestamp, float]] = {}
    charges: dict[str, dict[pd.Timestamp, float]] = {}
    warnings: list[str] = []
    rebalance_dates: list[pd.Timestamp] = []

    for step, start_index in enumerate(schedule):
        stop_index = boundaries[step + 1]
        when = dates[start_index]
        window = clean.iloc[start_index - config.lookback : start_index]

        if on_progress:
            on_progress(step / len(schedule), f"rebalancing {when.date()}")

        # One call per rebalance. Every strategy's drifted weights go in together so
        # the adapter can batch, or loop per strategy when turnover limits apply.
        previous = {
            name: dict(zip(assets, weights.tolist(), strict=True)) for name, weights in held.items()
        }
        try:
            frame, notes = optimizer(window, previous)
        except Exception as exc:  # noqa: BLE001 -- a failed re-optimisation holds the previous weights
            warnings.append(
                f"{when.date()}: optimizer failed ({type(exc).__name__}: {exc}); "
                "holding existing weights"
            )
            frame, notes = pd.DataFrame(index=assets), []

        warnings.extend(f"{when.date()}: {note}" for note in notes)

        if strategies is None:
            if frame.empty:
                raise ValueError("optimizer produced no strategies on the first rebalance")
            strategies = [str(column) for column in frame.columns]
            for name in strategies:
                gross[name], net[name] = [], []
                targets[name], turnovers[name], charges[name] = {}, {}, {}

        rebalance_dates.append(when)

        for name in strategies:
            drifted = held.get(name, np.zeros(len(assets)))
            if name in frame.columns:
                target = frame[name].reindex(assets).fillna(0.0).to_numpy(dtype=float)
            else:
                # Strategy failed this round; carry on with what is held rather than
                # dropping it from the comparison.
                target = drifted if drifted.any() else np.full(len(assets), 1.0 / len(assets))

            traded = float(np.abs(target - drifted).sum())
            charge = traded * cost_rate
            turnovers[name][when] = traded
            charges[name][when] = charge
            targets[name][when] = target.copy()
            held[name] = target.copy()

            for offset, day in enumerate(range(start_index, stop_index)):
                day_returns = values[day]
                period = float(held[name] @ day_returns)
                gross[name].append(period)
                # The cost lands once, on the day the trade happens.
                net[name].append(period - charge if offset == 0 else period)
                held[name] = _drift(held[name], day_returns, period)

    if on_progress:
        on_progress(1.0, "backtest complete")

    span = dates[schedule[0] :]
    return BacktestResult(
        net_returns=pd.DataFrame(net, index=span),
        gross_returns=pd.DataFrame(gross, index=span),
        weights={name: pd.DataFrame(history, index=assets).T for name, history in targets.items()},
        turnover=pd.DataFrame(turnovers),
        costs=pd.DataFrame(charges),
        rebalance_dates=rebalance_dates,
        warnings=warnings,
    )


def _drift(weights: np.ndarray, asset_returns: np.ndarray, portfolio_return: float) -> np.ndarray:
    """Weights after one period of holding, before any trade.

    ``w_i' = w_i (1 + r_i) / (1 + R)``. A portfolio that loses everything is reset
    to the drifted values unnormalised rather than dividing by zero.
    """
    grown = weights * (1.0 + asset_returns)
    total = 1.0 + portfolio_return
    return grown / total if abs(total) > 1e-12 else grown
