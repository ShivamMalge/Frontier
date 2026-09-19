"""Pydantic request/response contracts for the public API."""

from app.schemas.common import Frame, StrategyName, TickerError, WindowRequest
from app.schemas.forecast import (
    ForecastBackend,
    ForecastRequest,
    ForecastResponse,
    TickerForecastMetrics,
)
from app.schemas.jobs import JobAccepted, JobState, JobStatus
from app.schemas.market import PricesRequest, PricesResponse, ReturnsResponse
from app.schemas.meta import (
    HealthResponse,
    StrategyInfo,
    StrategyListResponse,
    UniverseResponse,
)
from app.schemas.pipeline import PipelineRequest, PipelineResult
from app.schemas.portfolio import (
    ConstraintSpec,
    FrontierPointOut,
    FrontierRequest,
    FrontierResponse,
    OptimizeRequest,
    OptimizeResponse,
    PerformanceRequest,
    PerformanceResponse,
    StrategyPerformance,
)

__all__ = [
    "ConstraintSpec",
    "Frame",
    "FrontierPointOut",
    "FrontierRequest",
    "FrontierResponse",
    "ForecastBackend",
    "ForecastRequest",
    "ForecastResponse",
    "HealthResponse",
    "JobAccepted",
    "JobState",
    "JobStatus",
    "OptimizeRequest",
    "OptimizeResponse",
    "PerformanceRequest",
    "PerformanceResponse",
    "PipelineRequest",
    "PipelineResult",
    "PricesRequest",
    "PricesResponse",
    "ReturnsResponse",
    "StrategyInfo",
    "StrategyListResponse",
    "StrategyName",
    "StrategyPerformance",
    "TickerError",
    "TickerForecastMetrics",
    "UniverseResponse",
    "WindowRequest",
]
