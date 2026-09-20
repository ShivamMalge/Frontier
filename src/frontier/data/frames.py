# src/frontier/data/frames.py

"""Conversions across the Polars/pandas boundary, and where that boundary sits.

Phase 5 makes the *data pipeline* Polars-native: loading prices, computing returns
and building features. It deliberately stops there. Two things downstream genuinely
need the other libraries:

* **cvxpy and the LSTM want numpy arrays.** Portfolio optimisation is linear algebra
  on an ``n x n`` covariance matrix; there is no dataframe work left to accelerate,
  and routing it through Polars would add conversions for no gain.
* **Riskfolio-Lib requires pandas.** Its API takes and returns ``pd.DataFrame``.

So the rule is: Polars from the data source up to the point where the numbers become
a matrix, numpy and pandas from there on. These helpers are the only sanctioned
crossings, which keeps the two libraries from leaking into each other's territory.

Polars has no index, so a wide frame carries its dates in a ``date`` column rather
than in an index. That is the main shape difference to keep in mind.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

DATE = "date"
TICKER = "ticker"
PRICE = "price"


def pandas_wide_to_polars(frame: pd.DataFrame, date_column: str = DATE) -> pl.DataFrame:
    """A date-indexed pandas frame becomes a Polars frame with a ``date`` column."""
    reset = frame.reset_index()
    # reset_index names an unnamed index "index"; whatever it is called, it is the
    # first column and it holds the dates.
    reset = reset.rename(columns={reset.columns[0]: date_column})
    return pl.from_pandas(reset)


def polars_wide_to_pandas(frame: pl.DataFrame, date_column: str = DATE) -> pd.DataFrame:
    """A Polars frame with a ``date`` column becomes a date-indexed pandas frame."""
    result = frame.to_pandas().set_index(date_column)
    result.index = pd.to_datetime(result.index)
    result.index.name = "Date"
    return result


def wide_to_long(frame: pl.DataFrame, value_name: str = PRICE) -> pl.DataFrame:
    """``date`` plus one column per ticker becomes ``date, ticker, <value>``."""
    return frame.unpivot(index=DATE, variable_name=TICKER, value_name=value_name).drop_nulls(
        value_name
    )


def long_to_wide(frame: pl.DataFrame, value_name: str = PRICE) -> pl.DataFrame:
    """``date, ticker, <value>`` becomes ``date`` plus one column per ticker."""
    return frame.pivot(on=TICKER, index=DATE, values=value_name).sort(DATE)


def series_to_numpy(frame: pl.DataFrame, columns: list[str]):
    """Column-major extraction for the model layer, in a fixed column order."""
    return frame.select(columns).to_numpy()
