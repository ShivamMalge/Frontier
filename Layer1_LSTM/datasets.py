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

Since Phase 5 the features for the whole universe are built in a single Polars pass
(:func:`build_splits`) rather than looping tickers in Python, and everything from
there on is numpy. pandas appears only at the caller's boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from Layer1_Preprocessing.feature_engineering import feature_names
from Layer1_Preprocessing.features_polars import build_features_all
from Layer1_Preprocessing.frames import pandas_wide_to_polars, wide_to_long

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


def build_splits(
    prices: pd.DataFrame,
    train_split: float,
    target: Target = "return",
    lags: int = 10,
    window: int | None = None,
    min_train_samples: int = 50,
) -> tuple[dict[str, SplitDataset], list[tuple[str, str]]]:
    """Build leak-free splits for every column of ``prices``.

    Features for the whole universe are computed in one Polars pass, then each
    ticker is split independently. Returns ``(splits, failures)`` so one bad ticker
    does not sink the batch.
    """
    names = feature_names(lags)
    long = wide_to_long(pandas_wide_to_polars(prices))
    featured = build_features_all(long, lags=lags)

    splits: dict[str, SplitDataset] = {}
    failures: list[tuple[str, str]] = []

    for ticker in prices.columns:
        label = str(ticker)
        rows = featured.filter(featured["ticker"] == label).sort("date")
        try:
            splits[label] = _assemble(
                ticker=label,
                dates=rows["date"].to_numpy(),
                price=rows["price"].to_numpy().astype(float),
                features=rows.select(names).to_numpy().astype(float),
                names=names,
                train_split=train_split,
                target=target,
                window=window,
                min_train_samples=min_train_samples,
            )
        except Exception as exc:  # noqa: BLE001 -- one unusable series must not sink the batch
            failures.append((label, f"{type(exc).__name__}: {exc}"))

    return splits, failures


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
    name = str(prices.name or "series")
    frame = prices.dropna().to_frame(name=name)
    splits, failures = build_splits(frame, train_split, target, lags, window, min_train_samples)
    if name not in splits:
        reason = failures[0][1] if failures else "no usable rows"
        raise ValueError(f"{name}: {reason}")
    return splits[name]


def _assemble(
    *,
    ticker: str,
    dates: np.ndarray,
    price: np.ndarray,
    features: np.ndarray,
    names: list[str],
    train_split: float,
    target: Target,
    window: int | None,
    min_train_samples: int,
) -> SplitDataset:
    """Turn one ticker's features into a chronologically split dataset."""
    if price.size < 2:
        raise ValueError("fewer than two observations")

    # The target on row i is realised on row i+1, so the last row has none.
    if target == "return":
        targets = np.concatenate([price[1:] / price[:-1] - 1.0, [np.nan]])
    else:
        targets = np.concatenate([price[1:], [np.nan]])
    actual_price = np.concatenate([price[1:], [np.nan]])
    target_dates = np.concatenate([dates[1:], dates[-1:]])

    usable = np.isfinite(targets) & np.isfinite(features).all(axis=1) & np.isfinite(price)
    if not usable.any():
        raise ValueError("no usable rows after feature warm-up")

    features = features[usable]
    targets = targets[usable]
    price_at_t = price[usable]
    actual_price = actual_price[usable]
    target_dates = pd.DatetimeIndex(target_dates[usable])

    split = int(np.ceil(len(targets) * train_split))
    if window is not None:
        split = max(split, window + 1)
    if split < min_train_samples or split >= len(targets):
        raise ValueError(
            f"train split of {train_split} leaves {split} training rows out of "
            f"{len(targets)}; need at least {min_train_samples} and some test rows"
        )

    # Scalers see the training slice only.
    scaled = np.column_stack(
        [
            Standardiser.fit(features[:split, column]).transform(features[:, column])
            for column in range(features.shape[1])
        ]
    )
    target_scaler = Standardiser.fit(targets[:split])
    y_scaled = target_scaler.transform(targets)

    if window is None:
        return SplitDataset(
            ticker=ticker,
            x_train=scaled[:split],
            y_train=y_scaled[:split],
            x_test=scaled[split:],
            y_test=y_scaled[split:],
            test_dates=target_dates[split:],
            test_previous_price=price_at_t[split:],
            test_actual_price=actual_price[split:],
            target_scaler=target_scaler,
            feature_names=names,
        )

    x_seq, y_seq, keep = _to_sequences(scaled, y_scaled, window)
    seq_split = int(np.searchsorted(keep, split))
    if seq_split <= 0 or seq_split >= len(y_seq):
        raise ValueError(f"a {window}-step window leaves no usable train/test split")

    return SplitDataset(
        ticker=ticker,
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


def _to_sequences(
    x: np.ndarray, y: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling windows of ``x`` ending at each row, aligned to ``y``.

    Uses a strided view rather than stacking slices in a Python loop. The loop
    version materialised every window: for 500 tickers at a 60-step window that is
    roughly 1.5 million windows of 60x22 floats, which took 40 seconds and several
    gigabytes. ``sliding_window_view`` returns a read-only view over the original
    buffer at no copy cost; PyTorch copies what it needs when building batches.
    """
    if len(y) <= window:
        raise ValueError(f"need more than {window} rows to build sequences, got {len(y)}")

    indices = np.arange(window - 1, len(y))
    # (n - window + 1, n_features, window) -> (n - window + 1, window, n_features).
    # Returned as a view: materialising it costs window x more memory than the
    # source, which for a large universe is the dominant cost of the whole pipeline.
    windows = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)
    return windows.transpose(0, 2, 1), y[indices], indices
