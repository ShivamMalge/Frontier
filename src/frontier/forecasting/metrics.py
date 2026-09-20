# src/frontier/forecasting/metrics.py

"""Forecast quality metrics.

A caution that applies to everything below computed on *price levels*: RMSE,
MAE, MAPE and R^2 are all scale-dependent and flattering on trending series. A
random-walk forecast (tomorrow == today) scores R^2 near 1.0 and MAPE near 1% on
daily equity closes while containing no information at all. Judge a model with
:func:`directional_accuracy` and :func:`mase_vs_naive`, which are both measured
against that baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def compute_basic_metrics(y_true, y_pred) -> tuple[float, float, float, float]:
    """Return ``(rmse, mae, r2, mape)`` on whatever units are supplied."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    safe_true = np.where(y_true == 0, 1e-8, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / safe_true)) * 100)
    return rmse, mae, r2, mape


def directional_accuracy(y_true, y_pred) -> float | None:
    """Share of *directional calls* that were correct. 0.5 is a coin flip.

    Compares ``y_pred[t] - y_true[t-1]`` against ``y_true[t] - y_true[t-1]``: did
    the model call the direction of the next move from the last observed price?

    Days where the model predicted no change are excluded rather than counted as
    wrong. A flat forecast expresses no directional opinion, and scoring it 0%
    reads as "always wrong" when the truth is "never guessed" -- the random walk
    baseline is exactly that case, and returns ``None`` here.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size < 2:
        return None

    actual_change = np.diff(y_true)
    predicted_change = y_pred[1:] - y_true[:-1]

    called = (actual_change != 0) & (predicted_change != 0)
    if not called.any():
        return None
    agree = np.sign(predicted_change[called]) == np.sign(actual_change[called])
    return float(agree.mean())


def mase_vs_naive(y_true, y_pred) -> float:
    """Mean absolute error relative to a random-walk forecast.

    Below 1.0 means the model beats "tomorrow == today"; at or above 1.0 it does
    not. This is the single number worth quoting for a price-level forecaster.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size < 2:
        return float("nan")

    model_error = np.mean(np.abs(y_pred[1:] - y_true[1:]))
    naive_error = np.mean(np.abs(y_true[:-1] - y_true[1:]))
    if naive_error == 0:
        return float("nan")
    return float(model_error / naive_error)


def approximate_accuracy(y_true, y_pred) -> float:
    """``100 * (1 - RMSE / mean(y_true))``.

    Retained only for continuity with earlier reports on this project. It is not
    a measure of forecast skill: it is scale-dependent (the same model scores
    higher on more expensive stocks) and a random walk typically scores above 97
    on it. Use :func:`mase_vs_naive` instead.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mean_actual = np.mean(y_true)
    if mean_actual == 0:
        mean_actual = 1e-8
    return float(np.clip(100 * (1 - rmse / mean_actual), 0.0, 100.0))
