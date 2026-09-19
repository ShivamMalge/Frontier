"""Shared fixtures.

Market data is stubbed so the suite never touches the network: yfinance is slow,
rate-limited, and silently revises history, none of which belongs in a test.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import jobs
from app.main import create_app
from app.services import market_data

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def synthetic_prices(tickers: list[str], periods: int = 900, seed: int = 11) -> pd.DataFrame:
    """Geometric random walks: realistic scale and autocorrelation, no network."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=periods)
    return pd.DataFrame(
        {
            ticker: 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.015, periods))
            for ticker in tickers
        },
        index=index,
    )


@pytest.fixture(autouse=True)
def _stub_market_data(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(tickers, start, end):  # noqa: ARG001 - signature match
        known = [t for t in tickers if t in TICKERS]
        if not known:
            return pd.DataFrame()
        return synthetic_prices(known)

    monkeypatch.setattr(market_data, "download_adjusted_close", fake_download)
    market_data.clear_cache()


@pytest.fixture(scope="session", autouse=True)
def _default_to_memory_backend() -> None:
    """Pin the default job backend for the suite.

    Without this every store construction would attempt localhost:6379, log a
    fallback warning and take the auto path. Tests that specifically exercise the
    Redis backend install their own store explicitly.
    """
    import os

    from app.settings import get_settings

    previous = os.environ.get("SO_JOB_BACKEND")
    os.environ["SO_JOB_BACKEND"] = "memory"
    get_settings.cache_clear()
    yield
    if previous is None:
        os.environ.pop("SO_JOB_BACKEND", None)
    else:
        os.environ["SO_JOB_BACKEND"] = previous
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_job_store() -> None:
    jobs.set_job_store(None)
    yield
    store = jobs.get_job_store()
    store.shutdown()
    jobs.set_job_store(None)


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def returns_frame() -> dict:
    """A returns Frame payload, as the API expects it on the wire."""
    from Layer1_Preprocessing.preprocessing import compute_returns

    returns = compute_returns(synthetic_prices(TICKERS, periods=400))
    return {
        "index": [d.date().isoformat() for d in returns.index],
        "columns": list(returns.columns),
        "data": returns.to_numpy().tolist(),
    }


@pytest.fixture
def window() -> dict:
    return {"start": dt.date(2018, 1, 1).isoformat(), "end": dt.date(2021, 8, 1).isoformat()}
