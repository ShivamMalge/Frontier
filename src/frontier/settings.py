"""Application configuration.

Domain defaults live in ``src/frontier/utils/config.py`` (framework-free, imported by the
numerical core). This module layers environment-variable overrides on top of
them, so the dependency direction is app -> utils and never the reverse.

Every field can be overridden with an ``SO_``-prefixed environment variable,
e.g. ``SO_RISK_FREE_RATE=0.04``.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from frontier.utils import config as domain


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SO_", env_file=".env", extra="ignore")

    app_name: str = "Frontier API"
    version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False

    # Phase 9 frontend runs on the Vite dev server.
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # --- domain defaults (see src/frontier/utils/config.py) ---
    universe: list[str] = domain.TECH_LIST
    start_date: str = domain.START_DATE
    end_date: str = domain.END_DATE
    risk_free_rate: float = domain.RISK_FREE_RATE
    lookback_window: int = domain.LOOKBACK_WINDOW
    train_split: float = domain.TRAIN_SPLIT
    lstm_epochs: int = domain.LSTM_EPOCHS
    lstm_batch_size: int = domain.LSTM_BATCH_SIZE
    trading_days_per_year: int = domain.TRADING_DAYS_PER_YEAR

    # --- service knobs ---
    #: "yfinance" hits the network. "synthetic" generates deterministic offline
    #: series. "parquet" reads the local store, which is the reproducible option:
    #: yfinance restates history, so only a stored snapshot gives repeatable results.
    market_data_source: Literal["yfinance", "synthetic", "parquet"] = "yfinance"
    #: Root of the Parquet price store.
    data_root: str = "data/store"
    default_forecast_backend: str = "naive"
    price_cache_ttl_seconds: int = 3600

    # --- tracking (Phase 7) ---
    #: Off by default. Tracking is telemetry: it never fails a run, and nothing is
    #: written unless this is set.
    mlflow_enabled: bool = False
    #: MLflow 3 deprecated the filesystem backend; SQLite is a single local file and
    #: needs no server. `mlflow ui --backend-store-uri <this>` to browse runs.
    mlflow_tracking_uri: str = "sqlite:///data/mlflow.db"
    mlflow_experiment: str = "frontier"

    # --- jobs (Phase 2) ---
    #: "auto" prefers Redis and falls back to the in-process store when it is
    #: unreachable; "redis" refuses to start without it; "memory" never uses it.
    #: Set SO_JOB_BACKEND=redis in production so a missing broker fails loudly.
    job_backend: Literal["auto", "redis", "memory"] = "auto"
    redis_url: str = "redis://localhost:6379/0"
    queue_name: str = "pipeline"
    #: Ceiling for one job. Training 18 LSTMs on CPU can take well over an hour.
    job_timeout_seconds: int = 7_200
    job_retention_seconds: int = 86_400
    redis_connect_timeout_seconds: float = 2.0
    redis_socket_timeout_seconds: float = 10.0
    #: Threads used only by the in-process store; RQ scales by process instead.
    job_workers: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()
