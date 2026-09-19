# Layer1_Preprocessing/data_loader.py

"""Adjusted-close price retrieval.

Downloads the whole universe in a single yfinance call rather than one call per
ticker. Besides being far faster, it returns a uniform MultiIndex frame that can
be sliced reliably -- the previous per-ticker concat produced a column index
whose shape depended on the yfinance version.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from utils.config import END_DATE, START_DATE
from utils.logger import log


def download_adjusted_close(
    tickers: list[str],
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """Return a date-indexed frame of adjusted closes, one column per ticker.

    Tickers with no data are omitted from the result; the caller compares the
    returned columns against what it requested to discover failures.
    """
    if not tickers:
        return pd.DataFrame()

    unique = list(dict.fromkeys(tickers))
    log(f"Downloading {len(unique)} tickers from {start} to {end}...")

    raw = yf.download(
        unique,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    field = "Adj Close" if "Adj Close" in _top_level(raw) else "Close"
    prices = raw[field] if isinstance(raw.columns, pd.MultiIndex) else raw[[field]]

    if isinstance(prices, pd.Series):
        # Single ticker: yfinance collapses the ticker level.
        prices = prices.to_frame(name=unique[0])

    prices = prices.copy()
    prices.columns = [str(column) for column in prices.columns]
    prices.index = pd.to_datetime(prices.index)

    # Drop tickers that came back entirely empty, then drop dates where any
    # surviving ticker is missing so every column shares one calendar.
    prices = prices.dropna(axis=1, how="all").sort_index()
    prices = prices.dropna(axis=0, how="any")

    missing = sorted(set(unique) - set(prices.columns))
    if missing:
        log(f"No data for: {', '.join(missing)}")

    return prices


def _top_level(frame: pd.DataFrame) -> list[str]:
    if isinstance(frame.columns, pd.MultiIndex):
        return [str(value) for value in frame.columns.get_level_values(0).unique()]
    return [str(column) for column in frame.columns]
