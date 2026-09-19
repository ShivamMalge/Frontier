"""Price store contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import WindowRequest


class IngestRequest(WindowRequest):
    """Download OHLCV and merge it into the Parquet store."""


class Revision(BaseModel):
    """A stored bar the provider has since restated."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    date: str
    old_adj_close: float
    new_adj_close: float


class IngestResponse(BaseModel):
    tickers: list[str]
    rows_written: int = Field(description="Rows in the store after the merge.")
    rows_added: int = Field(description="Dates not previously stored.")
    rows_revised: int = Field(
        description="Dates already stored whose values the provider has changed."
    )
    revised_tickers: list[str]
    revisions: list[Revision] = Field(
        default_factory=list, description="Up to 100 examples, newest last."
    )
    ingested_at: str


class TickerCoverage(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    rows: int
    first_date: str
    last_date: str
    last_ingested: str
    vintages: int = Field(description="Distinct ingestion batches contributing rows.")


class Gap(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    gap_start: str
    gap_end: str
    gap_days: int


class CoverageResponse(BaseModel):
    store_root: str
    exists: bool
    tickers: list[str]
    total_rows: int
    coverage: list[TickerCoverage]
    gaps: list[Gap] = Field(
        description="Consecutive stored dates more than max_gap_days apart. Weekends "
        "and holidays make small gaps normal; a large one means an interrupted "
        "ingestion or a delisting."
    )


class QueryRequest(BaseModel):
    """Run a read-only SQL query against the store."""

    model_config = ConfigDict(extra="forbid")

    sql: str = Field(
        min_length=1,
        max_length=4000,
        description="SELECT or WITH only. The store is exposed as a view named "
        "`prices` with columns date, ticker, open, high, low, close, adj_close, "
        "volume, ingested_at.",
    )
    limit: int = Field(default=1000, ge=1, le=100_000)


class QueryResponse(BaseModel):
    columns: list[str]
    rows: list[list[object]]
    row_count: int
    truncated: bool
