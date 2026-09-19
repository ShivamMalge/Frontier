# Layer1_LSTM/datasets.py

"""Supervised datasets for the forecasting backends.

One convention throughout, and it is the thing that keeps these models honest:

    features on row *t* use only data available at the close of day *t*,
    and the target is the move from *t* to *t + 1*.

So every sample answers "given everything known today, what happens tomorrow".
Nothing is shifted the wrong way, and no scaler ever sees the test window.

Two target choices:

``return``
    Predict tomorrow's simple return. This is the actual forecasting problem.
``price``
    Predict tomorrow's price level, as the original Keras pipeline did. Retained
    so the two can be compared directly -- a price-level model scores well on
    R^2 and MAPE while carrying almost no information, and being able to show
    that side by side is more useful than asserting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from Layer1_Preprocessing.feature_engineering import build_features

from .scaling import Standardiser

Target = Literal["return", "price"]


@dataclass(frozen=True)
class SplitDataset:
    """A chronologically split supervised dataset for one ticker."""

    ticker: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    #: Dates of the test targets -- i.e. the days being predicted.
    test_dates: pd.DatetimeIndex
    #: Actual close on the day *before* each test target, used to rebuild a price
    #: level from a predicted return.
    test_previous_price: np.ndarray
    #: Actual close on each test target date.
    test_actual_price: np.ndarray
    #: Fitted on the training targets only; inverts model output.
    target_scaler: Standardiser
    feature_names: list[str]

    @property
    def n_features(self) -> int:
        return self.x_train.shape[-1]

    def rebuild_prices(self, scaled_predictions: np.ndarray, target: Target) -> np.ndarray:
        """Turn scaled model output into predicted price levels."""
        raw = self.target_scaler.inverse(np.asarray(scaled_predictions, dtype=float).ravel())
        if target == "price":
            return raw
        # A predicted return is applied to the last *actual* observed close, which
        # is the information a one-step-ahead forecaster genuinely has.
        return self.test_previous_price * (1.0 + raw)


def build_split(
    prices: pd.Series,
    train_split: float,
    target: Target = "return",
    lags: int = 10,
    window: int | None = None,
    min_train_samples: int = 50,
) -> SplitDataset:
    """Build a leak-free split for one price series.

    ``window`` ``None`` produces a flat design matrix for tree models; an integer
    produces sequences of that length for recurrent models.
    """
    prices = prices.astype(float).dropna()
    features = build_features(prices, lags=lags)

    # Shifted back one step: the value on row t is what happens between t and t+1.
    targets = prices.pct_change().shift(-1) if target == "return" else prices.shift(-1)

    frame = features.copy()
    frame["__target__"] = targets
    frame["__price__"] = prices
    frame = frame.dropna()

    if frame.empty:
        raise ValueError(f"{prices.name}: no usable rows after feature warm-up")

    names = [column for column in frame.columns if not column.startswith("__")]
    x_all = frame[names].to_numpy(dtype=float)
    y_all = frame["__target__"].to_numpy(dtype=float)
    price_at_t = frame["__price__"].to_numpy(dtype=float)

    # The target on row i is realised on the following observation, so line up the
    # actual prices and dates accordingly.
    target_dates = _shift_index(frame.index)
    actual_price = np.concatenate([price_at_t[1:], [np.nan]])

    usable = ~np.isnan(actual_price)
    x_all, y_all = x_all[usable], y_all[usable]
    price_at_t, actual_price = price_at_t[usable], actual_price[usable]
    target_dates = target_dates[usable]

    split = int(np.ceil(len(y_all) * train_split))
    if window is not None:
        split = max(split, window + 1)
    if split < min_train_samples or split >= len(y_all):
        raise ValueError(
            f"{prices.name}: train split of {train_split} leaves {split} training rows "
            f"out of {len(y_all)}; need at least {min_train_samples} and some test rows"
        )

    # Scalers see the training slice only.
    feature_scalers = [Standardiser.fit(x_all[:split, column]) for column in range(x_all.shape[1])]
    x_scaled = np.column_stack(
        [scaler.transform(x_all[:, column]) for column, scaler in enumerate(feature_scalers)]
    )
    target_scaler = Standardiser.fit(y_all[:split])
    y_scaled = target_scaler.transform(y_all)

    if window is None:
        return SplitDataset(
            ticker=str(prices.name),
            x_train=x_scaled[:split],
            y_train=y_scaled[:split],
            x_test=x_scaled[split:],
            y_test=y_scaled[split:],
            test_dates=target_dates[split:],
            test_previous_price=price_at_t[split:],
            test_actual_price=actual_price[split:],
            target_scaler=target_scaler,
            feature_names=names,
        )

    x_seq, y_seq, keep = _to_sequences(x_scaled, y_scaled, window)
    # `keep` indexes the original rows that survived windowing.
    seq_split = int(np.searchsorted(keep, split))
    if seq_split <= 0 or seq_split >= len(y_seq):
        raise ValueError(
            f"{prices.name}: a {window}-step window leaves no usable train/test split"
        )

    return SplitDataset(
        ticker=str(prices.name),
        x_train=x_seq[:seq_split],
        y_train=y_seq[:seq_split],
        x_test=x_seq[seq_split:],
        y_test=y_seq[seq_split:],
        test_dates=target_dates[keep][seq_split:],
        test_previous_price=price_at_t[keep][seq_split:],
        test_actual_price=actual_price[keep][seq_split:],
        target_scaler=target_scaler,
        feature_names=names,
    )


def _shift_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Dates on which each row's target is realised: the next observation."""
    shifted = list(index[1:])
    # The final row has no following observation; it is dropped by the NaN filter.
    shifted.append(index[-1])
    return pd.DatetimeIndex(shifted)


def _to_sequences(
    x: np.ndarray, y: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack rolling windows of ``x`` ending at each row, aligned to ``y``."""
    if len(y) <= window:
        raise ValueError(f"need more than {window} rows to build sequences, got {len(y)}")
    indices = np.arange(window - 1, len(y))
    sequences = np.stack([x[i - window + 1 : i + 1] for i in indices])
    return sequences, y[indices], indices
