"""Market data endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.common import Frame, TickerError
from app.schemas.market import PricesRequest, PricesResponse, ReturnsResponse
from app.services import market_data

router = APIRouter(prefix="/market", tags=["market data"])


@router.post("/prices", response_model=PricesResponse, summary="Adjusted closing prices")
def prices(request: PricesRequest) -> PricesResponse:
    result = market_data.get_prices(request.tickers, request.start, request.end)
    return PricesResponse(
        start=result.start,
        end=result.end,
        tickers=result.tickers,
        failed=[
            TickerError(ticker=t, reason="no market data returned") for t in result.missing
        ],
        observations=len(result.prices),
        cached=result.cached,
        prices=Frame.from_pandas(result.prices),
    )


@router.post("/returns", response_model=ReturnsResponse, summary="Simple daily returns")
def returns(request: PricesRequest) -> ReturnsResponse:
    frame, result = market_data.get_returns(request.tickers, request.start, request.end)
    return ReturnsResponse(
        start=result.start,
        end=result.end,
        tickers=[str(column) for column in frame.columns],
        failed=[
            TickerError(ticker=t, reason="no market data returned") for t in result.missing
        ],
        observations=len(frame),
        cached=result.cached,
        returns=Frame.from_pandas(frame),
    )
