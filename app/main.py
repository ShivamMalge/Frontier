"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.errors import register_exception_handlers
from app.jobs import get_job_store, set_job_store
from app.routers import (
    backtest,
    data,
    forecast,
    market,
    meta,
    pipeline,
    portfolio,
    tracking,
)
from app.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

DESCRIPTION = """
Forecast equity prices, optimize portfolios across six strategies, and compare
their risk-adjusted performance.

**Units matter.** The forecasting models emit *price levels*; the optimizers
consume *returns*. `POST /forecast` returns both, and `predicted_returns` is the
field to feed into `POST /portfolio/optimize`.

Long runs go through `POST /pipeline/runs`, which returns a job to poll.
"""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Resolve the job backend at startup rather than on the first request, so a
    # misconfigured broker surfaces immediately and /health is accurate at once.
    store = get_job_store()
    logging.getLogger(__name__).info("job backend: %s", store.backend)
    yield
    store.shutdown()
    set_job_store(None)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=DESCRIPTION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    # /health sits at the root so probes need not track the API version.
    app.include_router(meta.health_router)
    for module in (
        meta,
        market,
        forecast,
        portfolio,
        pipeline,
        backtest,
        data,
        tracking,
    ):
        app.include_router(module.router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
