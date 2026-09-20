# src/frontier/forecasting/evaluate_lstm.py

"""Out-of-sample prediction for a trained LSTM."""

from __future__ import annotations

import numpy as np


def lstm_predict(model, scaler, stock_data, training_data_len, lookback_window):
    """Predict the held-out tail of ``stock_data``.

    Returns ``(predictions, actuals)`` in price units, both of length
    ``len(stock_data) - training_data_len``, or ``(None, None)`` when the test
    window is too short to form even one lookback sequence.
    """
    stock_data = np.asarray(stock_data, dtype=float).reshape(-1, 1)
    # The scaler was fitted on training data only, so test values may fall outside
    # [0, 1]. That is expected and is left uncorrected.
    scaled_data = np.asarray(scaler.transform(stock_data)).reshape(-1, 1)

    test_data = scaled_data[training_data_len - lookback_window :]
    x_test = [test_data[i - lookback_window : i, 0] for i in range(lookback_window, len(test_data))]
    if not x_test:
        return None, None

    x_test = np.asarray(x_test).reshape(-1, lookback_window, 1)
    y_test = stock_data[training_data_len:, 0]

    raw = np.asarray(model.predict(x_test, verbose=0)).reshape(-1, 1)
    return scaler.inverse(raw).ravel(), y_test.ravel()
