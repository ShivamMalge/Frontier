"""Experiment tracking endpoints.

MLflow's own UI is the richer interface (`mlflow ui --backend-store-uri ...`); these
exist so a client -- the Phase 9 front end included -- can list recent runs and
compare them without shelling out.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query

from app.schemas.tracking import TrackedRun, TrackedRunList, TrackingStatus
from app.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tracking", tags=["tracking"])


@router.get("/status", response_model=TrackingStatus, summary="Is tracking on, and where")
def status() -> TrackingStatus:
    settings = get_settings()
    if not settings.mlflow_enabled:
        return TrackingStatus(
            enabled=False,
            detail="set SO_MLFLOW_ENABLED=true to record runs",
        )

    base = TrackingStatus(
        enabled=True,
        tracking_uri=settings.mlflow_tracking_uri,
        experiment=settings.mlflow_experiment,
        ui_command=f"mlflow ui --backend-store-uri {settings.mlflow_tracking_uri}",
    )
    try:
        client, experiment_id = _client()
        if experiment_id is None:
            return base.model_copy(update={"run_count": 0})
        return base.model_copy(
            update={"run_count": len(client.search_runs([experiment_id], max_results=10_000))}
        )
    except Exception as exc:
        logger.warning("tracking store unreachable", exc_info=True)
        return base.model_copy(update={"detail": f"store unreachable: {exc}"})


@router.get("/runs", response_model=TrackedRunList, summary="Recent tracked runs")
def runs(
    limit: int = Query(default=25, ge=1, le=500),
    kind: str | None = Query(default=None, description="Filter to 'pipeline' or 'backtest'."),
) -> TrackedRunList:
    settings = get_settings()
    if not settings.mlflow_enabled:
        return TrackedRunList(runs=[], total=0)

    try:
        client, experiment_id = _client()
    except Exception:
        logger.warning("tracking store unreachable", exc_info=True)
        return TrackedRunList(runs=[], total=0)

    if experiment_id is None:
        return TrackedRunList(runs=[], total=0)

    filter_string = f"tags.kind = '{kind}'" if kind else ""
    found = client.search_runs(
        [experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=limit,
    )
    return TrackedRunList(runs=[_describe(run) for run in found], total=len(found))


def _client():
    """An MLflow client and the configured experiment's id, if it exists yet."""
    from mlflow.tracking import MlflowClient

    settings = get_settings()
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(settings.mlflow_experiment)
    return client, (experiment.experiment_id if experiment else None)


def _describe(run) -> TrackedRun:
    import datetime as dt

    def moment(milliseconds: int | None) -> str | None:
        if not milliseconds:
            return None
        return dt.datetime.fromtimestamp(milliseconds / 1000, dt.UTC).isoformat()

    tags = run.data.tags or {}
    return TrackedRun(
        run_id=run.info.run_id,
        run_name=tags.get("mlflow.runName"),
        status=run.info.status,
        started_at=moment(run.info.start_time),
        ended_at=moment(run.info.end_time),
        kind=tags.get("kind"),
        git_commit=tags.get("git_commit"),
        data_vintage=tags.get("data_vintage"),
        params=dict(run.data.params or {}),
        metrics={k: float(v) for k, v in (run.data.metrics or {}).items()},
    )
