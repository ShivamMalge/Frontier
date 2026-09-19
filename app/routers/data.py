"""Price store endpoints.

The store is what makes results reproducible: yfinance restates history, so a
pipeline that re-downloads on every run cannot repeat its own output. Ingest once,
then point `SO_MARKET_DATA_SOURCE=parquet` at the snapshot.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, Query, Response, status

from app.errors import ServiceError
from app.jobs import TaskRef, get_job_store
from app.schemas.data import (
    CoverageResponse,
    Gap,
    IngestRequest,
    QueryRequest,
    QueryResponse,
    TickerCoverage,
)
from app.schemas.jobs import JobAccepted
from app.services import market_data
from app.settings import get_settings
from app.tasks import RUN_INGEST

router = APIRouter(prefix="/data", tags=["price store"])

#: Only read-only statements are accepted. DuckDB will happily COPY to the local
#: filesystem or ATTACH another database, so this endpoint allows a single
#: SELECT/WITH and nothing else.
_READ_ONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(attach|copy|export|install|load|pragma|set|create|insert|update|delete|drop|"
    r"alter|call)\b",
    re.IGNORECASE,
)


class UnsafeQueryError(ServiceError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "unsafe_query"


@router.post(
    "/ingest",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Download prices into the store",
    description=(
        "Fetches full OHLCV and merges it into the Parquet store. Re-ingesting a "
        "window already held is idempotent, except that any bar the provider has "
        "restated since is reported as a revision rather than changed silently. "
        "Submitted as a job; poll the usual job endpoints."
    ),
)
def ingest(request: IngestRequest, response: Response) -> JobAccepted:
    prefix = get_settings().api_v1_prefix
    record = get_job_store().submit(
        TaskRef(RUN_INGEST, {"request": request.model_dump(mode="json")})
    )
    status_url = f"{prefix}/pipeline/runs/{record.job_id}"
    response.headers["Location"] = status_url
    return JobAccepted(
        job_id=record.job_id,
        state=record.state,
        status_url=status_url,
        result_url=f"{status_url}/result",
    )


@router.get(
    "/coverage",
    response_model=CoverageResponse,
    summary="What the store holds",
)
def coverage(
    max_gap_days: int = Query(default=5, ge=1, le=365),
) -> CoverageResponse:
    store = market_data.get_store()

    if not store.exists():
        return CoverageResponse(
            store_root=str(store.root),
            exists=False,
            tickers=[],
            total_rows=0,
            coverage=[],
            gaps=[],
        )

    summary = store.coverage()
    gaps = store.gaps(max_gap_days)

    return CoverageResponse(
        store_root=str(store.root),
        exists=True,
        tickers=store.stored_tickers(),
        total_rows=int(summary["rows"].sum()) if not summary.is_empty() else 0,
        coverage=[
            TickerCoverage(
                ticker=row["ticker"],
                rows=int(row["rows"]),
                first_date=str(row["first_date"]),
                last_date=str(row["last_date"]),
                last_ingested=str(row["last_ingested"]),
                vintages=int(row["vintages"]),
            )
            for row in summary.iter_rows(named=True)
        ],
        gaps=[
            Gap(
                ticker=row["ticker"],
                gap_start=str(row["gap_start"]),
                gap_end=str(row["gap_end"]),
                gap_days=int(row["gap_days"]),
            )
            for row in gaps.iter_rows(named=True)
        ],
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Read-only SQL against the store",
    description=(
        "Runs a single SELECT or WITH statement against the `prices` view. This is "
        "the reason DuckDB is here: coverage summaries, gap detection and ad-hoc "
        "questions are aggregations, which SQL states better than dataframe code."
    ),
)
def query(request: QueryRequest) -> QueryResponse:
    statement = request.sql.strip().rstrip(";")

    if not _READ_ONLY.match(statement):
        raise UnsafeQueryError("only SELECT and WITH statements are accepted")
    if _FORBIDDEN.search(statement):
        raise UnsafeQueryError(
            "the query contains a statement that can write or load data; DuckDB can "
            "reach the local filesystem, so only read-only queries are allowed"
        )
    if ";" in statement:
        raise UnsafeQueryError("only a single statement is accepted")

    store = market_data.get_store()
    frame = store.sql(f"SELECT * FROM ({statement}) AS q LIMIT {request.limit + 1}")

    truncated = frame.height > request.limit
    if truncated:
        frame = frame.head(request.limit)

    return QueryResponse(
        columns=frame.columns,
        rows=[[_plain(value) for value in row] for row in frame.iter_rows()],
        row_count=frame.height,
        truncated=truncated,
    )


def _plain(value: object) -> object:
    """Dates and timestamps become strings; everything else passes through."""
    import datetime as dt

    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    return value
