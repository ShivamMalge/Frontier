# Layer1_Preprocessing/preprocessing.py

"""Transformations between price levels and returns.

The distinction matters: the optimizers in Layer 2 assume *returns*. Feeding
them price levels yields expected returns and covariances that are orders of
magnitude too large, and performance figures that are meaningless.
"""

from __future__ import annotations

import pandas as pd


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
