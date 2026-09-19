# Layer1_LSTM/train_lstm.py

"""Legacy Keras LSTM training.

Superseded by ``Layer1_LSTM/torch_lstm.py`` in Phase 3. Retained because it is the
model the original project reported on, and keeping it runnable makes the
before/after comparison something you can execute rather than assert.

The scaler leakage flagged in Phase 1 is fixed here: normalisation is now fitted
on the training slice alone. Previously ``fit_transform`` ran on the full series,
so the test window's minimum and maximum set the scale used during training.
Metrics from this module were optimistic as a result, and are now honest --
expect them to look slightly worse than the original numbers, which is the point.

Everything else is unchanged: a single raw price feature, price-level target,
one model per ticker. Those are the parts ``torch_lstm`` improves on.
"""

from __future__ import annotations

import numpy as np

from utils.config import (
    LOOKBACK_WINDOW,
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    MIN_OBSERVATIONS,
    TRAIN_SPLIT,
)

from .lstm_model import build_lstm_model
from .scaling import MinMax


def make_sequences(scaled: np.ndarray, lookback_window: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice a scaled 1-column series into ``(X, y)`` supervised pairs."""
    x, y = [], []
    for i in range(lookback_window, len(scaled)):
        x.append(scaled[i - lookback_window : i, 0])
        y.append(scaled[i, 0])
    return np.asarray(x), np.asarray(y)


def train_lstm_model(
    stock_data: np.ndarray,
    lookback_window: int = LOOKBACK_WINDOW,
    train_split: float = TRAIN_SPLIT,
    epochs: int = LSTM_EPOCHS,
    batch_size: int = LSTM_BATCH_SIZE,
    verbose: int = 0,
):
    """Fit one LSTM on a single ``(n, 1)`` price series.

    Returns ``(model, scaler, training_data_len)``. The scaler exposes
    ``transform``/``inverse`` and is fitted on ``stock_data[:training_data_len]``.
    """
    stock_data = np.asarray(stock_data, dtype=float).reshape(-1, 1)
    if len(stock_data) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Not enough data for LSTM: need {MIN_OBSERVATIONS}, got {len(stock_data)}."
        )

    training_data_len = int(np.ceil(len(stock_data) * train_split))
    if training_data_len <= lookback_window:
        raise ValueError(
            f"train split of {train_split} leaves {training_data_len} rows, "
            f"which cannot fill a {lookback_window}-step lookback window."
        )

    # Fitted on the training slice only. Test prices above the training maximum
    # scale beyond 1.0, which is the honest consequence of a trending series.
    scaler = MinMax.fit(stock_data[:training_data_len])
    scaled_train = scaler.transform(stock_data[:training_data_len]).reshape(-1, 1)

    x_train, y_train = make_sequences(scaled_train, lookback_window)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    model = build_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

    return model, scaler, training_data_len
