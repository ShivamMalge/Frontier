# Layer1_LSTM/torch_lstm.py

"""PyTorch LSTM forecaster.

Replaces the Keras implementation. The architecture is deliberately the same
shape -- two recurrent layers into a small dense head, about 118k parameters --
so the comparison isolates what actually changed:

* **no scaler leakage.** Normalisation is fitted on the training slice alone.
* **multivariate input.** 22 engineered features per timestep instead of a single
  raw price, so the model has something to work with beyond the level itself.
* **returns as the default target.** Predicting tomorrow's price from a window of
  prices is close to an identity mapping; predicting tomorrow's *return* is the
  real problem.
* **early stopping on a chronological validation split** taken from the end of
  the training window, never a random shuffle -- shuffling time series validation
  leaks the future into model selection.
* **one model across the whole universe** when ``multi_series`` is set, instead of
  eighteen independent models. One training run, and the model can learn patterns
  that repeat across tickers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .datasets import SplitDataset, Target, build_splits
from .results import SeriesForecast

logger = logging.getLogger(__name__)


@dataclass
class TorchConfig:
    window: int = 60
    hidden_sizes: tuple[int, int] = (128, 64)
    dense_units: int = 25
    dropout: float = 0.1
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    #: Fraction of the *training* window held back for early stopping, taken
    #: chronologically from its end.
    validation_fraction: float = 0.15
    patience: int = 3
    seed: int = 42
    target: Target = "return"
    multi_series: bool = False
    lags: int = 10
    train_split: float = 2 / 3
    threads: int | None = None
    history: list[dict[str, float]] = field(default_factory=list)


def build_model(n_features: int, config: TorchConfig):
    """Two stacked LSTM layers into a dense head."""
    import torch
    from torch import nn

    class LSTMForecaster(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            first, second = config.hidden_sizes
            self.lstm1 = nn.LSTM(n_features, first, batch_first=True)
            self.lstm2 = nn.LSTM(first, second, batch_first=True)
            self.dropout = nn.Dropout(config.dropout)
            self.dense = nn.Linear(second, config.dense_units)
            self.head = nn.Linear(config.dense_units, 1)
            self.activation = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm1(x)
            out, _ = self.lstm2(out)
            last = out[:, -1, :]  # the step being predicted from
            last = self.dropout(last)
            return self.head(self.activation(self.dense(last))).squeeze(-1)

    return LSTMForecaster()


def _chronological_split(n: int, validation_fraction: float) -> tuple[slice, slice]:
    """Hold back the last ``validation_fraction`` of the training rows.

    Random validation on a time series lets the model be selected using future
    observations, which flatters it and does not survive live use.
    """
    cut = int(n * (1.0 - validation_fraction))
    cut = max(1, min(cut, n - 1)) if n > 1 else n
    return slice(0, cut), slice(cut, n)


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    config: TorchConfig,
    label: str = "model",
):
    """Fit one LSTM, returning ``(model, history)``."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if config.threads:
        torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_idx, val_idx = _chronological_split(len(y), config.validation_fraction)
    x_tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    y_tensor = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    loader = DataLoader(
        TensorDataset(x_tensor[train_idx], y_tensor[train_idx]),
        batch_size=config.batch_size,
        # Batches are shuffled, which is fine: each *sample* already carries its
        # own ordered window, and the train/validation boundary stays in time.
        shuffle=True,
        drop_last=False,
    )
    x_val, y_val = x_tensor[val_idx], y_tensor[val_idx]

    model = build_model(x.shape[-1], config)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    history: list[dict[str, float]] = []
    best_loss, best_state, stale = float("inf"), None, 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for batch_x, batch_y in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            # Recurrent nets on financial data produce occasional huge gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running += float(loss.detach()) * batch_x.shape[0]
            seen += batch_x.shape[0]

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val)) if len(y_val) else float("nan")

        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_loss": val_loss})

        if np.isfinite(val_loss) and val_loss < best_loss - 1e-9:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                logger.info("%s: early stop at epoch %d (best val %.6g)", label, epoch, best_loss)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict(model, x: np.ndarray) -> np.ndarray:
    import torch

    if len(x) == 0:
        return np.zeros(0)
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        return model(tensor).numpy().ravel()


def forecast_universe(
    prices: pd.DataFrame,
    config: TorchConfig,
    on_progress=None,
) -> tuple[list[SeriesForecast], list[tuple[str, str]]]:
    """Forecast every column of ``prices``.

    With ``multi_series`` one model is trained on the pooled sequences of all
    tickers; otherwise one model per ticker. Either way each ticker is scaled by
    its own training statistics, so pooling does not mix incompatible units.
    """
    # One Polars pass builds features for every ticker, rather than looping in
    # Python and rebuilding the same 22 columns per series.
    splits, failures = build_splits(
        prices,
        train_split=config.train_split,
        target=config.target,
        lags=config.lags,
        window=config.window,
    )

    if not splits:
        return [], failures

    forecasts: list[SeriesForecast] = []

    if config.multi_series:
        if on_progress:
            on_progress(0.1, f"training one model across {len(splits)} tickers")
        x = np.concatenate([split.x_train for split in splits.values()])
        y = np.concatenate([split.y_train for split in splits.values()])
        model, history = train_model(x, y, config, label="multi-series")
        config.history = history

        for position, (ticker, split) in enumerate(splits.items(), start=1):
            if on_progress:
                on_progress(0.6 + 0.4 * position / len(splits), f"predicting {ticker}")
            forecasts.append(_to_forecast(model, split, config))
        return forecasts, failures

    for position, (ticker, split) in enumerate(splits.items(), start=1):
        if on_progress:
            on_progress(position / len(splits), f"training {ticker}")
        try:
            model, history = train_model(split.x_train, split.y_train, config, label=ticker)
            config.history = history
            forecasts.append(_to_forecast(model, split, config))
        except Exception as exc:  # noqa: BLE001 -- a ticker that will not train is reported, not fatal
            failures.append((ticker, f"{type(exc).__name__}: {exc}"))

    return forecasts, failures


def _to_forecast(model, split: SplitDataset, config: TorchConfig) -> SeriesForecast:
    scaled = predict(model, split.x_test)
    return SeriesForecast(
        ticker=split.ticker,
        dates=split.test_dates,
        predicted=split.rebuild_prices(scaled, config.target),
        actual=split.test_actual_price,
    )
