# src/frontier/data/store.py

"""Parquet price store, queried with DuckDB.

Why this exists at all: **yfinance revises history**. Splits and dividends are
restated, bad ticks are corrected, and the adjusted close for a date five years ago
is not guaranteed to be the number it returned yesterday. A pipeline that
re-downloads on every run therefore cannot reproduce its own results, and a backtest
built that way is not repeatable -- which is a worse problem than it being slow.

So prices are ingested once into Parquet, partitioned by ticker, and every row
carries the ``ingested_at`` timestamp of the batch that wrote it. Re-ingesting the
same window **detects and reports** revisions rather than overwriting silently.

Division of labour, stated plainly because the two tools overlap:

* **Polars** does the bulk reads that feed the pipeline. ``scan_parquet`` pushes the
  ticker and date predicates into the file scan, and the result is already in the
  pipeline's native format, so there is nothing to convert.
* **DuckDB** does the SQL-shaped work: coverage summaries, gap detection, revision
  comparison, and any ad-hoc query a person wants to run against the store. These
  are aggregations and joins, which is what SQL is for and what would otherwise be
  hand-written dataframe code.

Neither is doing work the other could not. The split is about which one expresses
each job more clearly.
"""

from __future__ import annotations

import datetime as dt
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

#: Hive-partitioned so a single ticker can be rewritten without touching the rest,
#: and so both Polars and DuckDB can skip files that cannot match a filter.
PARTITION_COLUMN = "ticker"

SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "ticker": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "adj_close": pl.Float64,
    "volume": pl.Int64,
    "ingested_at": pl.Datetime("us"),
}

VALUE_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")


@dataclass(frozen=True)
class IngestReport:
    """What one ingestion actually changed."""

    tickers: list[str]
    rows_written: int
    rows_added: int
    rows_revised: int
    revisions: pl.DataFrame
    ingested_at: dt.datetime

    @property
    def revised_tickers(self) -> list[str]:
        if self.revisions.is_empty():
            return []
        return sorted(self.revisions["ticker"].unique().to_list())


class PriceStore:
    """A Parquet dataset of OHLCV bars with revision tracking."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------ paths

    @property
    def prices_root(self) -> Path:
        return self.root / "prices"

    def ticker_path(self, ticker: str) -> Path:
        return self.prices_root / f"{PARTITION_COLUMN}={ticker.upper()}" / "data.parquet"

    def exists(self) -> bool:
        return self.prices_root.exists() and any(self.prices_root.glob("*/*.parquet"))

    def stored_tickers(self) -> list[str]:
        if not self.prices_root.exists():
            return []
        prefix = f"{PARTITION_COLUMN}="
        return sorted(
            path.name[len(prefix) :]
            for path in self.prices_root.iterdir()
            if path.is_dir() and path.name.startswith(prefix)
        )

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)

    # ------------------------------------------------------------------ write

    def write(self, frame: pl.DataFrame, ingested_at: dt.datetime | None = None) -> IngestReport:
        """Merge ``frame`` into the store, reporting additions and revisions.

        A row is a *revision* when the same ``(ticker, date)`` already exists with a
        different value. The new value wins -- upstream corrections are usually
        genuine -- but the change is recorded so it is never silent.
        """
        stamp = ingested_at or dt.datetime.now(dt.UTC).replace(tzinfo=None)
        if frame.is_empty():
            return IngestReport([], 0, 0, 0, _empty_revisions(), stamp)

        incoming = _conform(frame).with_columns(
            pl.lit(stamp).cast(pl.Datetime("us")).alias("ingested_at")
        )
        tickers = sorted(incoming["ticker"].unique().to_list())

        revisions: list[pl.DataFrame] = []
        added = 0
        written = 0

        for ticker in tickers:
            fresh = incoming.filter(pl.col("ticker") == ticker).sort("date")
            path = self.ticker_path(ticker)

            if path.exists():
                existing = pl.read_parquet(path)
                changed = _detect_revisions(existing, fresh)
                if not changed.is_empty():
                    revisions.append(changed)
                    logger.info(
                        "%s: %d of %d rows revised upstream", ticker, changed.height, fresh.height
                    )
                added += fresh.height - _overlap(existing, fresh)
                merged = _merge(existing, fresh)
            else:
                added += fresh.height
                merged = fresh

            path.parent.mkdir(parents=True, exist_ok=True)
            # Partition value lives in the directory name, so drop the column.
            merged.drop(PARTITION_COLUMN).write_parquet(path, compression="zstd")
            written += merged.height

        combined = pl.concat(revisions) if revisions else _empty_revisions()
        return IngestReport(
            tickers=tickers,
            rows_written=written,
            rows_added=added,
            rows_revised=combined.height,
            revisions=combined,
            ingested_at=stamp,
        )

    # ------------------------------------------------------------------- read

    def scan(
        self,
        tickers: list[str] | None = None,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> pl.LazyFrame:
        """Lazy scan with predicates pushed into the Parquet read."""
        if not self.exists():
            return pl.LazyFrame(schema=SCHEMA)

        frame = pl.scan_parquet(self.prices_root / "**" / "*.parquet", hive_partitioning=True)
        if tickers:
            wanted = [ticker.upper() for ticker in tickers]
            frame = frame.filter(pl.col(PARTITION_COLUMN).is_in(wanted))
        if start is not None:
            frame = frame.filter(pl.col("date") >= start)
        if end is not None:
            frame = frame.filter(pl.col("date") < end)
        return frame.sort([PARTITION_COLUMN, "date"])

    def read(
        self,
        tickers: list[str] | None = None,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> pl.DataFrame:
        return self.scan(tickers, start, end).collect()

    def read_wide(
        self,
        tickers: list[str] | None = None,
        start: dt.date | None = None,
        end: dt.date | None = None,
        column: str = "adj_close",
    ) -> pl.DataFrame:
        """One column per ticker, for the forecasting pipeline."""
        long = self.read(tickers, start, end)
        if long.is_empty():
            return pl.DataFrame({"date": []}, schema={"date": pl.Date})
        return (
            long.select("date", PARTITION_COLUMN, column)
            .pivot(on=PARTITION_COLUMN, index="date", values=column)
            .sort("date")
        )

    # ------------------------------------------------------------- duckdb SQL

    def connect(self):
        """A DuckDB connection with the store registered as a view named ``prices``."""
        import duckdb

        connection = duckdb.connect()
        if not self.exists():
            connection.execute(
                "CREATE VIEW prices AS SELECT "
                "CAST(NULL AS DATE) AS date, CAST(NULL AS VARCHAR) AS ticker, "
                "CAST(NULL AS DOUBLE) AS adj_close, CAST(NULL AS BIGINT) AS volume, "
                "CAST(NULL AS TIMESTAMP) AS ingested_at WHERE FALSE"
            )
            return connection

        pattern = str(self.prices_root / "**" / "*.parquet").replace("'", "''")
        connection.execute(
            f"CREATE VIEW prices AS "
            f"SELECT * FROM read_parquet('{pattern}', hive_partitioning = true)"
        )
        return connection

    def sql(self, query: str) -> pl.DataFrame:
        """Run SQL against the ``prices`` view and return Polars."""
        with self.connect() as connection:
            return connection.sql(query).pl()

    def coverage(self) -> pl.DataFrame:
        """Per-ticker row counts, date span and most recent ingestion."""
        return self.sql(
            """
            SELECT ticker,
                   COUNT(*)              AS rows,
                   MIN(date)             AS first_date,
                   MAX(date)             AS last_date,
                   MAX(ingested_at)      AS last_ingested,
                   COUNT(DISTINCT ingested_at) AS vintages
            FROM prices
            GROUP BY ticker
            ORDER BY ticker
            """
        )

    def gaps(self, max_gap_days: int = 5) -> pl.DataFrame:
        """Consecutive stored dates more than ``max_gap_days`` apart.

        Weekends and holidays make small gaps normal; a large one means an
        interrupted ingestion or a delisting, and is worth knowing about before a
        backtest silently spans it.
        """
        return self.sql(
            f"""
            WITH ordered AS (
                SELECT ticker, date,
                       LAG(date) OVER (PARTITION BY ticker ORDER BY date) AS previous
                FROM prices
            )
            SELECT ticker, previous AS gap_start, date AS gap_end,
                   date - previous AS gap_days
            FROM ordered
            WHERE previous IS NOT NULL AND date - previous > {int(max_gap_days)}
            ORDER BY gap_days DESC, ticker, gap_start
            """
        )

    def vintages(self) -> pl.DataFrame:
        """Distinct ingestion timestamps and how many rows each wrote."""
        return self.sql(
            """
            SELECT ingested_at, COUNT(*) AS rows, COUNT(DISTINCT ticker) AS tickers
            FROM prices
            GROUP BY ingested_at
            ORDER BY ingested_at
            """
        )


# ---------------------------------------------------------------- internals


def _conform(frame: pl.DataFrame) -> pl.DataFrame:
    """Cast to the store's schema and drop rows without a usable price."""
    present = [name for name in SCHEMA if name in frame.columns and name != "ingested_at"]
    casts = [pl.col(name).cast(SCHEMA[name]) for name in present]
    return (
        frame.select(present)
        .with_columns(casts)
        .with_columns(pl.col("ticker").str.to_uppercase())
        .drop_nulls(["date", "ticker", "adj_close"])
        .unique(subset=["ticker", "date"], keep="last")
        .sort(["ticker", "date"])
    )


def _detect_revisions(existing: pl.DataFrame, fresh: pl.DataFrame) -> pl.DataFrame:
    """Rows whose stored values differ from the incoming ones for the same date."""
    stored = existing.select(["date", *[c for c in VALUE_COLUMNS if c in existing.columns]])
    incoming = fresh.select(["date", "ticker", *[c for c in VALUE_COLUMNS if c in fresh.columns]])
    joined = incoming.join(stored, on="date", how="inner", suffix="_old")

    shared = [c for c in VALUE_COLUMNS if c in incoming.columns and f"{c}_old" in joined.columns]
    if not shared or joined.is_empty():
        return _empty_revisions()

    differs = None
    for column in shared:
        # Compare with a relative tolerance: float round-trips through Parquet are
        # exact, but a genuine restatement is never a 1e-12 difference.
        condition = (
            (pl.col(column) - pl.col(f"{column}_old")).abs()
            > 1e-9 * pl.max_horizontal(pl.col(f"{column}_old").abs(), pl.lit(1.0))
        ).fill_null(False)
        differs = condition if differs is None else (differs | condition)

    changed = joined.filter(differs)
    if changed.is_empty():
        return _empty_revisions()

    return changed.select(
        pl.col("ticker"),
        pl.col("date"),
        pl.col("adj_close_old").alias("old_adj_close"),
        pl.col("adj_close").alias("new_adj_close"),
    ).sort(["ticker", "date"])


def _overlap(existing: pl.DataFrame, fresh: pl.DataFrame) -> int:
    return fresh.join(existing.select("date"), on="date", how="semi").height


def _merge(existing: pl.DataFrame, fresh: pl.DataFrame) -> pl.DataFrame:
    """Incoming rows win on conflict; untouched stored rows are preserved."""
    stored = existing.with_columns(pl.lit(fresh["ticker"][0]).alias(PARTITION_COLUMN))
    columns = [name for name in SCHEMA if name in fresh.columns]
    return (
        pl.concat([stored.select(columns), fresh.select(columns)], how="vertical")
        .unique(subset=["date"], keep="last")
        .sort("date")
    )


def _empty_revisions() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.Utf8,
            "date": pl.Date,
            "old_adj_close": pl.Float64,
            "new_adj_close": pl.Float64,
        }
    )
