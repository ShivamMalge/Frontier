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

from frontier import jobs
from frontier.api.main import create_app
from frontier.services import market_data

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
    def fake_download(tickers, start, end):
        known = [t for t in tickers if t in TICKERS]
        if not known:
            return pd.DataFrame()
        return synthetic_prices(known)

    monkeypatch.setattr(market_data, "download_adjusted_close", fake_download)
    market_data.clear_cache()


#: Settings the suite owns, whatever the surrounding environment says.
#:
#: ``SO_JOB_BACKEND`` -- without it every store construction would attempt
#: localhost:6379, log a fallback warning and take the auto path. Tests that
#: exercise the Redis backend install their own store explicitly.
#:
#: ``SO_MARKET_DATA_SOURCE`` -- ``_stub_market_data`` replaces the download
#: function, which only the ``yfinance`` branch of ``market_data._fetch`` calls.
#: Inheriting ``synthetic`` from the environment (CI exports it for the docker
#: and playwright jobs) would route around the stub and hand every ticker a
#: price series, including the unknown ones whose failure the API tests assert
#: on. Tests that want another source set it themselves with monkeypatch.setenv,
#: which still wins over this.
PINNED_ENVIRONMENT = {
    "SO_JOB_BACKEND": "memory",
    "SO_MARKET_DATA_SOURCE": "yfinance",
}


@pytest.fixture(scope="session", autouse=True)
def _pinned_environment() -> None:
    """Make the suite independent of the ambient environment."""
    import os

    from frontier.settings import get_settings

    previous = {key: os.environ.get(key) for key in PINNED_ENVIRONMENT}
    os.environ.update(PINNED_ENVIRONMENT)
    get_settings.cache_clear()
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
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
    from frontier.data.preprocessing import compute_returns

    returns = compute_returns(synthetic_prices(TICKERS, periods=400))
    return {
        "index": [d.date().isoformat() for d in returns.index],
        "columns": list(returns.columns),
        "data": returns.to_numpy().tolist(),
    }


@pytest.fixture
def window() -> dict:
    return {"start": dt.date(2018, 1, 1).isoformat(), "end": dt.date(2021, 8, 1).isoformat()}
