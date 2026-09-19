# Layer1_LSTM/lstm_model.py

"""The LSTM architecture: two recurrent layers into a small dense head.

Roughly 118k parameters. Phase 3 reimplements this in PyTorch and adds a
multi-series variant that trains one model across the whole universe rather than
one model per ticker.
"""

from __future__ import annotations


def build_lstm_model(input_shape: tuple[int, int]):
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(128, return_sequences=True),
            LSTM(64, return_sequences=False),
            Dense(25),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
