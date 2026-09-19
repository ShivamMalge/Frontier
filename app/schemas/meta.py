"""Service metadata contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import StrategyName


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    version: str
    forecast_backends: list[str] = Field(description="Backends available in this process.")
    job_backend: str = Field(
        description="Job store actually in use: 'redis' or 'memory'. Reported so that a "
        "fallback from Redis to the in-process store is never silent."
    )


class BackendInfo(BaseModel):
    name: str
    scope: str = Field(description="'ticker' trains per series; 'universe' can pool them.")
    available: bool = Field(description="False when the required library is not installed.")
    requires: str | None = None
    description: str


class BackendListResponse(BaseModel):
    backends: list[BackendInfo]


class UniverseResponse(BaseModel):
    """Defaults a client needs in order to build a valid request."""

    tickers: list[str]
    default_start: str
    default_end: str
    lookback_window: int
    train_split: float
    risk_free_rate: float
    trading_days_per_year: int


class StrategyInfo(BaseModel):
    name: StrategyName
    label: str = Field(description="Human-readable name for display.")
    family: str = Field(
        description="Broad category: mean-variance, risk-based, hierarchical, "
        "robust-covariance or tail-risk."
    )
    solver: str = Field(description="Library that solves it: 'cvxpy' or 'riskfolio'.")
    long_only: bool = Field(description="Whether weights are constrained to be non-negative.")
    uses_expected_returns: bool = Field(
        description="False for risk-only optimizers, which ignore the return forecast. "
        "True for exactly one of the ten strategies."
    )
    respects_constraints: bool = Field(
        description="False where the strategy's own construction determines every weight, "
        "leaving no freedom for caller-supplied bounds; such requests return a warning."
    )
    description: str


class StrategyListResponse(BaseModel):
    strategies: list[StrategyInfo]
