# Layer1_Preprocessing/preprocessing.py

"""Transformations between price levels and returns.

The distinction matters: the optimizers in Layer 2 assume *returns*. Feeding
them price levels yields expected returns and covariances that are orders of
magnitude too large, and performance figures that are meaningless.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import polars as pl


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a frame of price levels."""
    if prices.empty:
        return prices
    return prices.pct_change().dropna(how="all")


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns, for when additive aggregation over time is wanted."""
    import numpy as np

    if prices.empty:
        return prices
    return np.log(prices / prices.shift(1)).dropna(how="all")


def compute_returns_polars(prices: pl.DataFrame, date_column: str = "date") -> pl.DataFrame:
    """Simple daily returns for a wide Polars frame of price levels.

    Every non-date column is differenced in one pass. The first row is dropped,
    matching the pandas version's ``dropna``.
    """
    import polars as pl

    tickers = [column for column in prices.columns if column != date_column]
    if not tickers:
        return prices

    return (
        prices.sort(date_column)
        .with_columns([pl.col(ticker).pct_change().alias(ticker) for ticker in tickers])
        .slice(1)
    )


def compute_returns_long(
    prices: pl.DataFrame, date_column: str = "date", value: str = "price"
) -> pl.DataFrame:
    """Simple daily returns for a long Polars frame of ``date, ticker, price``."""
    import polars as pl

    return (
        prices.sort(["ticker", date_column])
        .with_columns(pl.col(value).pct_change().over("ticker").alias("return"))
        .drop_nulls("return")
    )
