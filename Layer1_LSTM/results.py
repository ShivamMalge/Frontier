# Layer1_LSTM/results.py

"""Result type shared by every forecasting backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import (
    approximate_accuracy,
    compute_basic_metrics,
    directional_accuracy,
    mase_vs_naive,
)


@dataclass(frozen=True)
class SeriesForecast:
    """One ticker's out-of-sample forecast, in price units.

    Every backend reports price levels regardless of what it modelled internally,
    so that metrics are comparable across backends. A model trained on returns
    reconstructs the level as ``previous_actual_price * (1 + predicted_return)``.
    """

    ticker: str
    dates: pd.DatetimeIndex
    predicted: np.ndarray
    actual: np.ndarray

    def __post_init__(self) -> None:
        if not (len(self.dates) == self.predicted.size == self.actual.size):
            raise ValueError(
                f"{self.ticker}: dates/predicted/actual lengths disagree "
                f"({len(self.dates)}, {self.predicted.size}, {self.actual.size})"
            )

    def metrics(self) -> dict[str, float]:
        rmse, mae, r2, mape = compute_basic_metrics(self.actual, self.predicted)
        return {
            "observations": int(self.actual.size),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "directional_accuracy": directional_accuracy(self.actual, self.predicted),
            "mase_vs_naive": mase_vs_naive(self.actual, self.predicted),
            "legacy_approximate_accuracy": approximate_accuracy(self.actual, self.predicted),
        }
