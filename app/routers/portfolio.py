"""Optimization and performance endpoints.

Both are stateless: the caller supplies the return series. That keeps them fast
and lets a client re-optimize against edited forecasts without re-running one.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter

from app.schemas.common import Frame
from app.schemas.portfolio import (
    OptimizeRequest,
    OptimizeResponse,
    PerformanceRequest,
    PerformanceResponse,
    StrategyPerformance,
)
from app.services import optimization, performance
from app.settings import get_settings

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


def _to_pandas(frame: Frame) -> pd.DataFrame:
    return pd.DataFrame(frame.data, index=frame.index, columns=frame.columns, dtype=float)


@router.post(
    "/optimize",
    response_model=OptimizeResponse,
    summary="Weights for each optimization strategy",
)
def optimize(request: OptimizeRequest) -> OptimizeResponse:
    settings = get_settings()
    returns = _to_pandas(request.returns)
    strategies = [s.value for s in request.strategies] if request.strategies else None

    weights, warnings = optimization.optimize(
        returns, strategies, request.risk_free_rate
    )
    return OptimizeResponse(
        tickers=[str(t) for t in weights.index],
        observations=len(returns),
        risk_free_rate=(
            settings.risk_free_rate
            if request.risk_free_rate is None
            else request.risk_free_rate
        ),
        weights=Frame.from_pandas(weights),
        warnings=warnings,
    )


@router.post(
    "/performance",
    response_model=PerformanceResponse,
    summary="Risk and return statistics per strategy",
)
def evaluate(request: PerformanceRequest) -> PerformanceResponse:
    settings = get_settings()
    returns = _to_pandas(request.returns)
    weights = _to_pandas(request.weights)

    perf, portfolio_returns, cumulative = performance.evaluate(
        returns, weights, request.risk_free_rate
    )
    return PerformanceResponse(
        risk_free_rate=(
            settings.risk_free_rate
            if request.risk_free_rate is None
            else request.risk_free_rate
        ),
        performance=[
            StrategyPerformance(
                strategy=str(name),
                annual_return=float(row["Annual Return"]),
                annual_volatility=float(row["Annual Vol"]),
                sharpe=float(row["Sharpe"]),
                sortino=float(row["Sortino"]),
                max_drawdown=float(row["Max Drawdown"]),
            )
            for name, row in perf.iterrows()
        ],
        portfolio_returns=Frame.from_pandas(portfolio_returns),
        cumulative_growth=Frame.from_pandas(cumulative),
    )
