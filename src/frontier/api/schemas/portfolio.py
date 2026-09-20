"""Optimization and performance contracts."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from frontier.api.schemas.common import Frame, StrategyName


class ConstraintSpec(BaseModel):
    """Convex constraints on a fully-invested portfolio.

    Weights always sum to 1; everything here narrows the feasible set further.
    Contradictions are reported as a 422 before any solver runs.

    Not every strategy can honour these: risk parity, HRP and the Gerber variants
    determine every weight from their own construction, so a request carrying extra
    bounds comes back with a warning naming them. `GET /meta/strategies` reports
    which strategies respect constraints.
    """

    model_config = ConfigDict(extra="forbid")

    min_weight: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Negative permits short positions."
    )
    max_weight: float = Field(default=1.0, gt=0.0, le=1.0)
    max_leverage: float | None = Field(
        default=None,
        ge=1.0,
        le=5.0,
        description="Cap on gross exposure, sum |w|. 1.0 forbids shorting; 1.3 is a "
        "130/30 mandate. Only meaningful alongside a negative min_weight.",
    )
    group_caps: dict[str, tuple[list[str], float]] = Field(
        default_factory=dict,
        description='Named group ceilings, e.g. {"tech": [["AAPL", "MSFT"], 0.4]}.',
    )
    group_floors: dict[str, tuple[list[str], float]] = Field(
        default_factory=dict, description="Named group floors, same shape as group_caps."
    )
    max_turnover: float | None = Field(
        default=None,
        ge=0.0,
        le=4.0,
        description="Cap on sum |w - w_previous|. Requires previous_weights.",
    )
    previous_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Current holdings, for turnover. Omitted names count as zero.",
    )

    @model_validator(mode="after")
    def _check_coherent(self) -> Self:
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight cannot exceed max_weight")
        if self.max_turnover is not None and not self.previous_weights:
            raise ValueError("max_turnover requires previous_weights")
        return self

    def to_domain(self):
        from frontier.optimization.constraints import Constraints

        return Constraints(
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            max_leverage=self.max_leverage,
            group_caps={k: (list(v[0]), v[1]) for k, v in self.group_caps.items()},
            group_floors={k: (list(v[0]), v[1]) for k, v in self.group_floors.items()},
            max_turnover=self.max_turnover,
            previous_weights=dict(self.previous_weights),
        )


class OptimizeRequest(BaseModel):
    """Optimize against a caller-supplied return series.

    ``returns`` must be *returns*, not price levels. Supplying prices produces
    covariances and expected returns that are orders of magnitude too large and
    renders every downstream performance figure meaningless.
    """

    model_config = ConfigDict(extra="forbid")

    returns: Frame
    strategies: list[StrategyName] | None = Field(
        default=None, description="Defaults to every available strategy."
    )
    risk_free_rate: float | None = Field(default=None, ge=-0.5, le=1.0)
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)

    @model_validator(mode="after")
    def _reject_price_levels(self) -> Self:
        """Guard against the price-vs-returns mix-up.

        Daily simple returns live within a few percent. A mean absolute value
        above 1.0 (i.e. +/-100% per day) means prices were passed by mistake.
        """
        values = [abs(v) for row in self.returns.data for v in row if v is not None]
        if values and sum(values) / len(values) > 1.0:
            raise ValueError(
                "mean absolute value exceeds 1.0, which suggests price levels were supplied "
                "instead of returns; pass simple returns (e.g. 0.012 for +1.2%)"
            )
        return self


class OptimizeResponse(BaseModel):
    tickers: list[str]
    observations: int
    risk_free_rate: float
    weights: Frame = Field(description="Rows are tickers, columns are strategy names.")
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues, e.g. a solver that did not converge.",
    )


class PerformanceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: Frame = Field(description="Asset returns, one column per ticker.")
    weights: Frame = Field(description="Rows are tickers, columns are strategy names.")
    risk_free_rate: float | None = Field(default=None, ge=-0.5, le=1.0)


class StrategyPerformance(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: str
    annual_return: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float


class PerformanceResponse(BaseModel):
    risk_free_rate: float
    performance: list[StrategyPerformance]
    portfolio_returns: Frame = Field(description="Per-strategy portfolio return series.")
    cumulative_growth: Frame = Field(description="Compounded growth of 1.0 unit invested.")


class FrontierRequest(BaseModel):
    """Trace the efficient frontier for a set of returns."""

    model_config = ConfigDict(extra="forbid")

    returns: Frame
    points: int = Field(default=40, ge=2, le=200)
    risk_free_rate: float | None = Field(default=None, ge=-0.5, le=1.0)
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)


class FrontierPointOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    expected_return: float = Field(description="Annualised.")
    volatility: float = Field(description="Annualised.")
    sharpe: float
    weights: dict[str, float]


class FrontierResponse(BaseModel):
    tickers: list[str]
    observations: int
    risk_free_rate: float
    points: list[FrontierPointOut] = Field(description="Ordered by expected return.")
    tangency_index: int = Field(description="Index of the highest-Sharpe point.")
