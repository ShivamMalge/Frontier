# Layer1_Preprocessing/synthetic.py

"""Deterministic synthetic price series.

An offline stand-in for the market data provider, for development, demos,
reproducible examples and any test that runs in a separate process and so cannot
be handed a fixture. Each ticker is seeded from its own name, so the same symbol
always produces the same history.

These are geometric random walks. They have realistic scale, drift and
volatility, and no predictable structure whatsoever -- which makes them a useful
sanity check: a forecasting model that appears to beat a random walk on this data
is measuring a bug, not skill.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

ANNUAL_DRIFT = 0.08
ANNUAL_VOLATILITY = 0.24
TRADING_DAYS = 252


def _seed_for(ticker: str) -> int:
    digest = hashlib.sha256(ticker.upper().encode()).digest()
    return int.from_bytes(digest[:8], "big")


def synthetic_prices(
    tickers: list[str],
    start: str,
    end: str,
    base_price: float = 100.0,
) -> pd.DataFrame:
    """A date-indexed frame of synthetic closes, one column per ticker."""
    index = pd.bdate_range(start=start, end=end)
    if len(index) == 0:
        return pd.DataFrame()

    daily_drift = ANNUAL_DRIFT / TRADING_DAYS
    daily_vol = ANNUAL_VOLATILITY / np.sqrt(TRADING_DAYS)

    columns = {}
    for ticker in dict.fromkeys(t.upper() for t in tickers):
        rng = np.random.default_rng(_seed_for(ticker))
        shocks = rng.normal(daily_drift, daily_vol, len(index))
        # Spread starting prices across a plausible range, per ticker.
        start_price = base_price * (0.5 + (_seed_for(ticker) % 1000) / 500.0)
        columns[ticker] = start_price * np.cumprod(1.0 + shocks)

    return pd.DataFrame(columns, index=index)
