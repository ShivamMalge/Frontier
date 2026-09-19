# Layer1_Preprocessing/data_loader.py

"""Price retrieval from the upstream provider.

Downloads the whole universe in a single yfinance call rather than one call per
ticker. Besides being far faster, that returns a uniform frame which can be sliced
reliably -- the original per-ticker concat produced a column index whose shape
depended on the yfinance version.

Since Phase 6 the full OHLCV record is fetched, not just adjusted closes. Volume and
intraday range were previously unavailable to the feature builder for no better
reason than the download being narrow.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import polars as pl
import yfinance as yf

from utils.config import END_DATE, START_DATE
from utils.logger import log

#: Canonical column order for a long OHLCV frame.
OHLCV_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]

_RENAME = {
    "Date": "date",
    "Ticker": "ticker",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def download_ohlcv(
    tickers: list[str],
    start: str = START_DATE,
    end: str = END_DATE,
) -> pl.DataFrame:
    """Return a long Polars frame of ``date, ticker, open..volume``.

    Tickers with no data are simply absent from the result; callers compare the
    returned tickers against what they asked for to discover failures.
    """
    if not tickers:
        return pl.DataFrame(schema=_empty_schema())

    unique = list(dict.fromkeys(ticker.upper() for ticker in tickers))
    log(f"Downloading OHLCV for {len(unique)} tickers from {start} to {end}...")

    raw = yf.download(
        unique, start=start, end=end, auto_adjust=False, progress=False, group_by="column"
    )
    if raw is None or raw.empty:
        return pl.DataFrame(schema=_empty_schema())

    long = _to_long(raw, unique)
    if long.is_empty():
        return long

    missing = sorted(set(unique) - set(long["ticker"].unique().to_list()))
    if missing:
        log(f"No data for: {', '.join(missing)}")
    return long


def _to_long(raw: pd.DataFrame, tickers: list[str]) -> pl.DataFrame:
    """Flatten yfinance's ``(Price, Ticker)`` column MultiIndex into long rows."""
    if isinstance(raw.columns, pd.MultiIndex):
        flat = raw.stack(level="Ticker", future_stack=True).reset_index()
    else:
        # Single ticker: yfinance collapses the ticker level entirely.
        flat = raw.reset_index()
        flat["Ticker"] = tickers[0]

    flat = flat.rename(columns=_RENAME)
    if "adj_close" not in flat.columns and "close" in flat.columns:
        # Newer yfinance versions can omit Adj Close; fall back rather than fail.
        flat["adj_close"] = flat["close"]

    keep = [column for column in OHLCV_COLUMNS if column in flat.columns]
    flat = flat[keep].dropna(subset=["adj_close"])

    frame = pl.from_pandas(flat)
    return _normalise(frame)


def _normalise(frame: pl.DataFrame) -> pl.DataFrame:
    casts = [
        pl.col("date").cast(pl.Date),
        pl.col("ticker").cast(pl.Utf8).str.to_uppercase(),
        *[
            pl.col(name).cast(pl.Float64)
            for name in ("open", "high", "low", "close", "adj_close")
            if name in frame.columns
        ],
    ]
    if "volume" in frame.columns:
        casts.append(pl.col("volume").cast(pl.Int64))
    return frame.with_columns(casts).sort(["ticker", "date"])


def _empty_schema() -> dict[str, object]:
    return {
        "date": pl.Date,
        "ticker": pl.Utf8,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "adj_close": pl.Float64,
        "volume": pl.Int64,
    }


def download_adjusted_close(
    tickers: list[str],
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Adjusted closes as a date-indexed pandas frame, one column per ticker.

    The pandas-shaped view the service layer has used since Phase 1, now derived
    from the full OHLCV download.
    """
    from .frames import long_to_wide, polars_wide_to_pandas

    long = download_ohlcv(tickers, start, end)
    if long.is_empty():
        return pd.DataFrame()

    wide = long_to_wide(long.rename({"adj_close": "price"}), value_name="price")
    prices = polars_wide_to_pandas(wide)
    # Keep one shared calendar across tickers, as before.
    return prices.dropna(axis=1, how="all").sort_index().dropna(axis=0, how="any")


def as_date(value: str | dt.date) -> dt.date:
    return value if isinstance(value, dt.date) else dt.date.fromisoformat(value)
