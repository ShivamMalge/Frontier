# Layer1_Preprocessing/features_polars.py

"""Feature engineering in Polars, computed for every ticker in one pass.

The pandas implementation in ``feature_engineering.py`` builds 22 columns for one
price series, and callers loop over tickers in Python. This module takes the whole
universe in long format and expresses each feature as a window over ``ticker``, so
the loop happens once, inside Rust, across every series at the same time.

Two reasons that is worth doing beyond raw speed:

* **No index to misalign.** Polars has no index. Several of the original defects in
  this project were index-alignment hazards -- a frame silently reindexed, a
  ``droplevel`` that removed the wrong level, columns assigned positionally. A
  column named ``ticker`` cannot be quietly reindexed behind your back.
* **Lazy evaluation.** The whole pipeline is one query plan that Polars optimises
  and parallelises, rather than 22 eager passes per ticker.

Column names and semantics match ``feature_engineering.build_features`` exactly, and
a test asserts the two agree to floating-point tolerance. The convention is
unchanged: every feature on row *t* uses only data available at the close of *t*.
"""

from __future__ import annotations

import polars as pl

from .feature_engineering import LONG_WINDOW, MEDIUM_WINDOW, SHORT_WINDOW, feature_names
from .frames import DATE, PRICE, TICKER, long_to_wide, wide_to_long

__all__ = [
    "build_features_all",
    "build_features_lazy",
    "long_to_wide",
    "wide_to_long",
]


#: Intermediate columns computed once and reused. Recomputing ``pct_change`` inside
#: each feature expression -- as a naive translation does -- repeats it thirteen
#: times per ticker and dominates the runtime.
_RETURN = "__ret"
_DELTA = "__delta"


def _base_columns() -> list[pl.Expr]:
    return [
        pl.col(PRICE).pct_change().over(TICKER).alias(_RETURN),
        pl.col(PRICE).diff().over(TICKER).alias(_DELTA),
    ]


def _return_lags(lags: int) -> list[pl.Expr]:
    # Lag 1 is the return realised at t, which is known at the close of t.
    return [
        pl.col(_RETURN).shift(lag - 1).over(TICKER).alias(f"ret_lag_{lag}")
        for lag in range(1, lags + 1)
    ]


def _volatility() -> list[pl.Expr]:
    return [
        pl.col(_RETURN)
        .rolling_std(window_size=window, ddof=1)
        .over(TICKER)
        .alias(f"vol_{window}")
        for window in (SHORT_WINDOW, MEDIUM_WINDOW, LONG_WINDOW)
    ]


def _momentum() -> list[pl.Expr]:
    return [
        pl.col(PRICE).pct_change(window).over(TICKER).alias(f"mom_{window}")
        for window in (SHORT_WINDOW, MEDIUM_WINDOW, LONG_WINDOW)
    ]


def _moving_average_gap() -> list[pl.Expr]:
    return [
        (
            pl.col(PRICE) / pl.col(PRICE).rolling_mean(window_size=window).over(TICKER) - 1.0
        ).alias(f"ma_gap_{window}")
        for window in (MEDIUM_WINDOW, LONG_WINDOW)
    ]


def _range_position() -> pl.Expr:
    low = pl.col(PRICE).rolling_min(window_size=MEDIUM_WINDOW).over(TICKER)
    high = pl.col(PRICE).rolling_max(window_size=MEDIUM_WINDOW).over(TICKER)
    return ((pl.col(PRICE) - low) / (high - low)).alias("range_position")


def _rsi(window: int = 14) -> pl.Expr:
    """Wilder's RSI scaled to [0, 1].

    Wilder's smoothing is an EWM with ``alpha = 1/window`` and no startup
    adjustment, which is what ``adjust=False`` selects.
    """
    gain = (
        pl.col(_DELTA)
        .clip(lower_bound=0.0)
        .ewm_mean(alpha=1.0 / window, adjust=False)
        .over(TICKER)
    )
    loss = (
        (-pl.col(_DELTA).clip(upper_bound=0.0))
        .ewm_mean(alpha=1.0 / window, adjust=False)
        .over(TICKER)
    )
    # A zero average loss leaves RSI undefined; pandas propagates NaN and the caller
    # fills 0.5, so match that.
    strength = gain / pl.when(loss == 0.0).then(None).otherwise(loss)
    return (1.0 - 1.0 / (1.0 + strength)).fill_null(0.5).alias("rsi_14")


def _calendar() -> list[pl.Expr]:
    return [
        # Polars weekday is 1=Monday; pandas dayofweek is 0=Monday.
        (pl.col(DATE).dt.weekday() - 1).cast(pl.Float64).alias("weekday"),
        pl.col(DATE).dt.month().cast(pl.Float64).alias("month"),
    ]


def build_features_lazy(prices: pl.LazyFrame, lags: int = 10) -> pl.LazyFrame:
    """Feature expressions over a long frame of ``date, ticker, price``.

    Returns a lazy frame; nothing is computed until it is collected, so Polars can
    fuse and parallelise the whole pipeline.
    """
    return (
        prices.sort([TICKER, DATE])
        .with_columns(*_base_columns())
        .with_columns(
            *_return_lags(lags),
            *_volatility(),
            *_momentum(),
            *_moving_average_gap(),
            _range_position(),
            _rsi(),
            *_calendar(),
        )
        .with_columns(
            # JSON has no infinity, and a zero-width range or moving average yields it.
            [
                pl.when(pl.col(name).is_infinite())
                .then(None)
                .otherwise(pl.col(name))
                .alias(name)
                for name in feature_names(lags)
            ]
        )
        .drop(_RETURN, _DELTA)
    )


def build_features_all(prices: pl.DataFrame, lags: int = 10) -> pl.DataFrame:
    """Eager wrapper over :func:`build_features_lazy`."""
    return build_features_lazy(prices.lazy(), lags).collect()
