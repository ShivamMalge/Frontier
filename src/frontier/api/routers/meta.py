"""Health and service-metadata endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from frontier.api.schemas.meta import (
    BackendInfo,
    BackendListResponse,
    HealthResponse,
    StrategyInfo,
    StrategyListResponse,
    UniverseResponse,
)
from frontier.jobs import get_job_store
from frontier.services.forecasting import available_backends, backend_specs
from frontier.services.optimization import STRATEGY_INFO
from frontier.settings import get_settings

#: Mounted at the root so that probes do not need to know the API version.
health_router = APIRouter(tags=["meta"])

#: Mounted under the versioned prefix.
router = APIRouter(prefix="/meta", tags=["meta"])


@health_router.get("/health", response_model=HealthResponse, summary="Liveness and capabilities")
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        app=settings.app_name,
        version=settings.version,
        forecast_backends=available_backends(),
        job_backend=get_job_store().backend,
    )


@router.get(
    "/universe",
    response_model=UniverseResponse,
    summary="Default universe and pipeline parameters",
)
def universe() -> UniverseResponse:
    settings = get_settings()
    return UniverseResponse(
        tickers=settings.universe,
        default_start=settings.start_date,
        default_end=settings.end_date,
        lookback_window=settings.lookback_window,
        train_split=settings.train_split,
        risk_free_rate=settings.risk_free_rate,
        trading_days_per_year=settings.trading_days_per_year,
    )


@router.get(
    "/strategies",
    response_model=StrategyListResponse,
    summary="Available optimization strategies",
)
def strategies() -> StrategyListResponse:
    return StrategyListResponse(
        strategies=[
            StrategyInfo(name=name, **info)  # type: ignore[arg-type]
            for name, info in STRATEGY_INFO.items()
        ]
    )


@router.get(
    "/backends",
    response_model=BackendListResponse,
    summary="Available forecasting backends",
)
def backends() -> BackendListResponse:
    return BackendListResponse(
        backends=[
            BackendInfo(
                name=spec.name,
                scope=spec.scope,
                available=spec.available,
                requires=spec.requires,
                description=spec.description,
            )
            for spec in backend_specs()
        ]
    )
