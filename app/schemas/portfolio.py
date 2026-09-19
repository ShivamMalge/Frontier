"""Optimization and performance contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.common import Frame, StrategyName


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

    @model_validator(mode="after")
    def _reject_price_levels(self) -> OptimizeRequest:
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
