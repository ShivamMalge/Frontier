"""Shared contract primitives."""

from __future__ import annotations

import datetime as dt
import math
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Accepts either case; :class:`WindowRequest` upper-cases on the way in.
Ticker = Annotated[
    str,
    Field(min_length=1, max_length=12, pattern=r"^[A-Za-z0-9.\-]+$"),
]


class StrategyName(StrEnum):
    """Optimizer outputs. The first six values match the historical CSV columns."""

    MARKOWITZ_MAX_SHARPE = "Markowitz_MaxSharpe"
    MARKOWITZ_MIN_VAR = "Markowitz_MinVar"
    RISK_PARITY = "RiskParity"
    GMV = "GMV"
    HRP = "HRP"
    GERBER_INV_VAR = "Gerber_InvVar"
    # Added in Phase 4, once Riskfolio-Lib made them a few lines each.
    HRP_CVAR = "HRP_CVaR"
    GERBER_HRP = "Gerber_HRP"
    MIN_CVAR = "MinCVaR"
    MIN_CDAR = "MinCDaR"


class Frame(BaseModel):
    """A 2-D labelled table in column-oriented wire format.

    Chosen over a list of row objects because it does not repeat the column
    names on every one of several thousand rows, and because it maps directly
    onto both ``pandas.DataFrame`` and the ECharts/Plotly dataset format that
    Phase 9 will consume.

    ``data`` is row-major: ``data[i][j]`` is the value at ``index[i]`` for
    ``columns[j]``. ``None`` encodes a missing observation.

    Row labels are strings so that one type covers both time series (ISO-8601
    dates, e.g. a return series) and cross-sectional tables (tickers, e.g. a
    weights matrix).
    """

    model_config = ConfigDict(frozen=True)

    index: list[str] = Field(description="Row labels: ISO-8601 dates, or tickers.")
    columns: list[str] = Field(description="Column labels (tickers or strategy names).")
    data: list[list[float | None]] = Field(description="Row-major values.")

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        width = len(self.columns)
        for position, row in enumerate(self.data):
            if len(row) != width:
                raise ValueError(
                    f"row {position} has {len(row)} values but there are {width} columns"
                )
        if len(self.data) != len(self.index):
            raise ValueError(
                f"data has {len(self.data)} rows but index has {len(self.index)} labels"
            )
        return self

    @classmethod
    def from_pandas(cls, frame: object) -> Frame:
        """Build from a ``pandas.DataFrame``, mapping NaN/inf to ``None``.

        Imported lazily so that the schema module stays cheap to import.
        """
        import pandas as pd

        if not isinstance(frame, pd.DataFrame):  # pragma: no cover - defensive
            raise TypeError(f"expected a DataFrame, got {type(frame).__name__}")

        return cls(
            index=[_as_label(label) for label in frame.index],
            columns=[str(column) for column in frame.columns],
            data=[[_finite(value) for value in row] for row in frame.to_numpy().tolist()],
        )


    @classmethod
    def from_polars(cls, frame: object, index_column: str = "date") -> Frame:
        """Build from a ``polars.DataFrame`` whose row labels live in a column.

        Polars has no index, so the labels are an ordinary column. Avoids a
        round trip through pandas purely to serialise.
        """
        import polars as pl

        if not isinstance(frame, pl.DataFrame):  # pragma: no cover - defensive
            raise TypeError(f"expected a polars DataFrame, got {type(frame).__name__}")
        if index_column not in frame.columns:
            raise ValueError(f"index column {index_column!r} not in {frame.columns}")

        columns = [name for name in frame.columns if name != index_column]
        return cls(
            index=[_as_label(value) for value in frame[index_column].to_list()],
            columns=columns,
            data=[
                [_finite(value) for value in row]
                for row in frame.select(columns).iter_rows()
            ],
        )


def _as_label(label: object) -> str:
    """Dates become ISO-8601 strings; anything else becomes its str()."""
    if isinstance(label, dt.datetime):
        return label.date().isoformat()
    if isinstance(label, dt.date):
        return label.isoformat()
    return str(label)


def _finite(value: object) -> float | None:
    """JSON has no NaN or Infinity; represent both as null."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class WindowRequest(BaseModel):
    """Ticker selection plus an observation window."""

    model_config = ConfigDict(extra="forbid")

    tickers: list[Ticker] = Field(min_length=1, max_length=100)
    start: dt.date | None = Field(default=None, description="Defaults to the configured start.")
    end: dt.date | None = Field(default=None, description="Defaults to the configured end.")

    @model_validator(mode="after")
    def _normalise(self) -> Self:
        if self.start and self.end and self.start >= self.end:
            raise ValueError("start must be strictly before end")
        # Upper-case and de-duplicate while preserving the caller's order.
        self.tickers = list(dict.fromkeys(t.upper() for t in self.tickers))
        return self


class TickerError(BaseModel):
    """A single ticker that could not be processed, and why."""

    ticker: str
    reason: str
