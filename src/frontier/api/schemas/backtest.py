"""Walk-forward backtest contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from frontier.api.schemas.common import Frame, StrategyName, WindowRequest
from frontier.api.schemas.forecast import ForecastBackend
from frontier.api.schemas.portfolio import ConstraintSpec


class BacktestRequest(WindowRequest):
    """Re-optimise on a trailing window and hold forward, charging for trades.

    Unlike `POST /portfolio/optimize`, nothing here is fitted on the data it is
    scored against: at each rebalance the optimizer sees only the preceding
    `lookback` observations.
    """

    strategies: list[StrategyName] | None = Field(
        default=None, description="Defaults to every strategy."
    )
    lookback: int = Field(
        default=252,
        ge=20,
        le=2520,
        description="Trailing observations used to estimate returns and covariance.",
    )
    rebalance_every: int = Field(
        default=21,
        ge=1,
        le=252,
        description="Trading days between rebalances. 21 is roughly monthly.",
    )
    cost_bps: float = Field(
        default=10.0,
        ge=0.0,
        le=500.0,
        description="Cost in basis points per unit of traded notional, charged on "
        "sum |dw| at each rebalance.",
    )
    risk_free_rate: float | None = Field(default=None, ge=-0.5, le=1.0)
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)
    on_forecast: ForecastBackend | None = Field(
        default=None,
        description="Reserved: backtest against a forecast rather than realised "
        "returns. Not yet implemented; realised returns are always used.",
    )


class BacktestStrategyResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: str
    annual_return: float = Field(description="Net of costs.")
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    gross_annual_return: float = Field(description="Before costs.")
    cost_drag: float = Field(description="Gross annual return minus net.")
    avg_turnover: float = Field(description="Mean sum |dw| per rebalance.")
    annual_turnover: float = Field(description="Turnover scaled to a year.")
    total_cost: float = Field(description="Sum of all costs, as a fraction of value.")
    rebalances: int


class BacktestResponse(BaseModel):
    tickers: list[str]
    observations: int = Field(description="Out-of-sample days evaluated.")
    lookback: int
    rebalance_every: int
    cost_bps: float
    rebalance_dates: list[str]
    results: list[BacktestStrategyResult]
    cumulative_growth: Frame = Field(description="Net of costs, 1.0 invested.")
    gross_cumulative_growth: Frame
    turnover: Frame = Field(description="Rows are rebalance dates, columns strategies.")
    warnings: list[str] = Field(default_factory=list)
    tracking_run_id: str | None = Field(
        default=None, description="MLflow run id when tracking is enabled."
    )
