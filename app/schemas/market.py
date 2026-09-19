"""Market data contracts."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field

from app.schemas.common import Frame, TickerError, WindowRequest


class PricesRequest(WindowRequest):
    pass


class _MarketResponse(BaseModel):
    start: dt.date
    end: dt.date
    tickers: list[str] = Field(description="Tickers successfully retrieved.")
    failed: list[TickerError] = Field(default_factory=list)
    observations: int
    cached: bool = Field(description="True when served from the in-process price cache.")


class PricesResponse(_MarketResponse):
    """Adjusted closing prices."""

    prices: Frame


class ReturnsResponse(_MarketResponse):
    """Simple daily returns derived from adjusted closing prices."""

    returns: Frame
