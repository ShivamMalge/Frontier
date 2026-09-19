"""Price retrieval with an in-process cache.

The cache is deliberately simple. Phase 6 replaces it with Parquet on disk
queried through DuckDB, which also makes backtests reproducible -- yfinance
silently revises history, so today's download is not guaranteed to match
yesterday's.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from dataclasses import dataclass

import pandas as pd

from app.errors import NoMarketDataError
from app.settings import get_settings
from Layer1_Preprocessing.data_loader import download_adjusted_close
from Layer1_Preprocessing.preprocessing import compute_returns
from Layer1_Preprocessing.synthetic import synthetic_prices


@dataclass(frozen=True)
class PriceResult:
    prices: pd.DataFrame
    start: dt.date
    end: dt.date
    requested: list[str]
    cached: bool

    @property
    def tickers(self) -> list[str]:
        return [str(column) for column in self.prices.columns]

    @property
    def missing(self) -> list[str]:
        return sorted(set(self.requested) - set(self.tickers))


class PriceCache:
    """Thread-safe TTL cache keyed by (tickers, window)."""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str, str], tuple[float, pd.DataFrame]] = {}

    def get(self, key: tuple[str, str, str]) -> pd.DataFrame | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, frame = entry
            if time.monotonic() - stored_at > self._ttl:
                del self._entries[key]
                return None
            return frame.copy()

    def put(self, key: tuple[str, str, str], frame: pd.DataFrame) -> None:
        with self._lock:
            self._entries[key] = (time.monotonic(), frame.copy())

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_cache = PriceCache(get_settings().price_cache_ttl_seconds)


def _fetch(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Dispatch to the configured data source.

    ``synthetic`` needs no network, which makes it the right choice for offline
    development and for tests whose work runs in a separate process. Phase 6 adds
    a ``parquet`` source backed by DuckDB.
    """
    source = get_settings().market_data_source
    if source == "synthetic":
        return synthetic_prices(tickers, start, end)
    return download_adjusted_close(tickers, start, end)


def get_prices(
    tickers: list[str],
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> PriceResult:
    """Adjusted closes for ``tickers``, from cache when available."""
    settings = get_settings()
    start = start or dt.date.fromisoformat(settings.start_date)
    end = end or dt.date.fromisoformat(settings.end_date)

    requested = list(dict.fromkeys(ticker.upper() for ticker in tickers))
    key = (",".join(sorted(requested)), start.isoformat(), end.isoformat())

    cached_frame = _cache.get(key)
    if cached_frame is not None:
        return PriceResult(cached_frame, start, end, requested, cached=True)

    prices = _fetch(requested, start.isoformat(), end.isoformat())
    if prices.empty:
        raise NoMarketDataError(
            "the upstream data provider returned no rows for this request",
            tickers=",".join(requested),
            start=start,
            end=end,
        )

    _cache.put(key, prices)
    return PriceResult(prices, start, end, requested, cached=False)


def get_returns(
    tickers: list[str],
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> tuple[pd.DataFrame, PriceResult]:
    """Simple daily returns, alongside the price result they came from."""
    result = get_prices(tickers, start, end)
    return compute_returns(result.prices), result


def clear_cache() -> None:
    _cache.clear()
