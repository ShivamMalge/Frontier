"""Backtest endpoints.

Submitted as a job rather than run inline: a five-year monthly backtest across ten
strategies is several hundred separate optimisations, which is minutes of work.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from frontier.api.schemas.backtest import BacktestRequest
from frontier.api.schemas.jobs import JobAccepted
from frontier.jobs import TaskRef, get_job_store
from frontier.settings import get_settings
from frontier.tasks import RUN_BACKTEST

router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.post(
    "/runs",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a walk-forward backtest",
    description=(
        "Re-optimises on a trailing window at every rebalance and holds forward "
        "through unseen data, charging `cost_bps` on traded notional. Weights drift "
        "with prices between rebalances rather than being pinned to their targets. "
        "Poll the same job endpoints as a pipeline run."
    ),
)
def submit(request: BacktestRequest, response: Response) -> JobAccepted:
    prefix = get_settings().api_v1_prefix
    record = get_job_store().submit(
        TaskRef(RUN_BACKTEST, {"request": request.model_dump(mode="json")})
    )

    status_url = f"{prefix}/pipeline/runs/{record.job_id}"
    response.headers["Location"] = status_url
    return JobAccepted(
        job_id=record.job_id,
        state=record.state,
        status_url=status_url,
        result_url=f"{status_url}/result",
    )
