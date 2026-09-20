# src/frontier/forecasting/scaling.py

"""Normalisation fitted on training data only.

The original pipeline called ``MinMaxScaler().fit_transform(full_series)`` before
splitting, so the minimum and maximum of the *test* window set the scale used
during training. That is lookahead: the model was told, indirectly, how high and
low the series would go. Every scaler here is fitted on the training slice alone,
and test values are free to fall outside the fitted range -- which is the honest
outcome, not a defect to clip away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Standardiser:
    """Zero-mean, unit-variance scaling fitted on a training slice."""

    mean: float
    scale: float

    @classmethod
    def fit(cls, train: np.ndarray) -> Standardiser:
        train = np.asarray(train, dtype=float).ravel()
        mean = float(np.mean(train))
        scale = float(np.std(train))
        # A constant series has no spread; fall back to 1.0 so transform is a shift.
        return cls(mean=mean, scale=scale if scale > 1e-12 else 1.0)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.mean


@dataclass(frozen=True)
class MinMax:
    """Min-max scaling to [0, 1], fitted on a training slice.

    Test values outside the training range map outside [0, 1]. That is expected
    for a series that trends: a model trained on 2010-2018 prices genuinely has
    not seen 2023 levels, and pretending otherwise is the leak this replaces.
    """

    low: float
    span: float

    @classmethod
    def fit(cls, train: np.ndarray) -> MinMax:
        train = np.asarray(train, dtype=float).ravel()
        low = float(np.min(train))
        span = float(np.max(train)) - low
        return cls(low=low, span=span if span > 1e-12 else 1.0)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.low) / self.span

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.span + self.low
