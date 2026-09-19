# Layer1_Preprocessing/feature_engineering.py

"""Features for the tree-based and neural forecasters.

This module was a placeholder returning its input unchanged, which meant every
model saw nothing but a window of raw prices. These features are all computed
from the *past* of a single series -- each row uses only data available strictly
before the value being predicted, so nothing here leaks the future.

Volume and cross-sectional features are deliberately absent: the data loader
currently fetches adjusted closes only. Adding volume means widening the download,
which is a Phase 6 concern once prices are stored in Parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Lookbacks used by the rolling features, in trading days.
SHORT_WINDOW = 5
MEDIUM_WINDOW = 21
LONG_WINDOW = 63


def build_features(prices: pd.Series, lags: int = 10) -> pd.DataFrame:
    """Per-date features for one price series.

    Returns a frame indexed like ``prices`` with NaN in the warm-up rows; callers
    drop them together with the target.
    """
    prices = prices.astype(float)
    returns = prices.pct_change()

    features: dict[str, pd.Series] = {}

    # Recent returns: the raw autoregressive signal.
    for lag in range(1, lags + 1):
        features[f"ret_lag_{lag}"] = returns.shift(lag - 1)

    # Realised volatility over several horizons.
    for window in (SHORT_WINDOW, MEDIUM_WINDOW, LONG_WINDOW):
        features[f"vol_{window}"] = returns.rolling(window).std()

    # Mean reversion / momentum: cumulative return over each horizon.
    for window in (SHORT_WINDOW, MEDIUM_WINDOW, LONG_WINDOW):
        features[f"mom_{window}"] = prices.pct_change(window)

    # Position relative to a moving average, scale-free.
    for window in (MEDIUM_WINDOW, LONG_WINDOW):
        moving_average = prices.rolling(window).mean()
        features[f"ma_gap_{window}"] = prices / moving_average - 1.0

    # Where today sits in its recent range: 0 at the low, 1 at the high.
    rolling_low = prices.rolling(MEDIUM_WINDOW).min()
    rolling_high = prices.rolling(MEDIUM_WINDOW).max()
    features["range_position"] = (prices - rolling_low) / (rolling_high - rolling_low)

    features["rsi_14"] = relative_strength_index(prices, 14)

    # Day-of-week and month, for the calendar effects that do show up in equities.
    features["weekday"] = pd.Series(prices.index.dayofweek, index=prices.index, dtype=float)
    features["month"] = pd.Series(prices.index.month, index=prices.index, dtype=float)

    frame = pd.DataFrame(features, index=prices.index)
    return frame.replace([np.inf, -np.inf], np.nan)


def relative_strength_index(prices: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI, scaled to [0, 1] rather than [0, 100]."""
    delta = prices.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / window, adjust=False).mean()
    relative_strength = gain / loss.replace(0.0, np.nan)
    return (1.0 - 1.0 / (1.0 + relative_strength)).fillna(0.5)


def feature_names(lags: int = 10) -> list[str]:
    """Column order produced by :func:`build_features`, without computing it."""
    names = [f"ret_lag_{lag}" for lag in range(1, lags + 1)]
    names += [f"vol_{w}" for w in (SHORT_WINDOW, MEDIUM_WINDOW, LONG_WINDOW)]
    names += [f"mom_{w}" for w in (SHORT_WINDOW, MEDIUM_WINDOW, LONG_WINDOW)]
    names += [f"ma_gap_{w}" for w in (MEDIUM_WINDOW, LONG_WINDOW)]
    names += ["range_position", "rsi_14", "weekday", "month"]
    return names


def add_features(df):
    """Deprecated alias kept so older callers keep working.

    The original signature took and returned a frame unchanged. Use
    :func:`build_features`, which takes a single price series.
    """
    return df
