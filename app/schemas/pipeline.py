"""End-to-end pipeline contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import Frame, StrategyName, TickerError
from app.schemas.forecast import ForecastRequest, TickerForecastMetrics
from app.schemas.portfolio import ConstraintSpec, StrategyPerformance


class PipelineRequest(ForecastRequest):
    """Forecast, optimize, construct and score in one submission."""

    strategies: list[StrategyName] | None = None
    risk_free_rate: float | None = Field(default=None, ge=-0.5, le=1.0)
    risk_tolerance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="0 selects the lowest-volatility strategy, 1 the highest.",
    )
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)


class PipelineResult(BaseModel):
    tickers: list[str]
    failed: list[TickerError] = Field(default_factory=list)
    backend: str
    forecast_metrics: list[TickerForecastMetrics]
    weights: Frame
    performance: list[StrategyPerformance]
    cumulative_growth: Frame
    selected_strategy: str
    selected_weights: dict[str, float]
    warnings: list[str] = Field(default_factory=list)
    diagnostics: dict[str, object] = Field(
        default_factory=dict,
        description="Backend extras, e.g. LightGBM feature importances.",
    )
