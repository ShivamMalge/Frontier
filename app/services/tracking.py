"""Experiment tracking with MLflow.

The reason this phase exists: the original project reported "93%+ accuracy across
all stocks" with no record of what was measured, on which data, by which metric, or
against what baseline. The number could not be reproduced or argued with. Every run
now logs its parameters, its data vintage, its seed and its metrics -- including the
baseline comparison and the legacy metric side by side, so the discrepancy is
visible rather than something you have to be told about.

Two design rules:

**Tracking never breaks the work.** A run is telemetry. If the tracking store is
unreachable or a metric fails to serialise, the pipeline logs a warning and carries
on. Losing a log entry is a nuisance; failing a forty-minute backtest because of one
is not.

**Disabled by default.** ``SO_MLFLOW_ENABLED=true`` turns it on. Nothing here fires
otherwise, and the no-op context manager keeps the call sites free of conditionals.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import logging
import os
import subprocess
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from app.settings import get_settings

logger = logging.getLogger(__name__)

#: MLflow prints a hint about its tracing skill on import; irrelevant here, and it
#: pollutes worker logs.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

#: MLflow rejects metric names containing most punctuation.
_SAFE = str.maketrans({".": "_", " ": "_", "/": "_", "-": "_", ":": "_"})


@dataclass
class Run:
    """A handle to one tracked run, or a no-op when tracking is disabled."""

    active: bool = False
    run_id: str | None = None
    _params: dict[str, Any] = field(default_factory=dict)
    _metrics: dict[str, float] = field(default_factory=dict)

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self.active:
            return
        cleaned = {
            str(key): _stringify(value) for key, value in params.items() if value is not None
        }
        self._params.update(cleaned)
        _guard("log_params", lambda: _mlflow().log_params(cleaned))

    def log_metrics(self, metrics: Mapping[str, Any], prefix: str = "") -> None:
        if not self.active:
            return
        cleaned: dict[str, float] = {}
        for key, value in metrics.items():
            number = _as_float(value)
            if number is None:
                # None is meaningful here -- directional accuracy is null for a flat
                # forecast -- and MLflow has no representation for it, so skip rather
                # than log a misleading zero.
                continue
            cleaned[f"{prefix}{str(key).translate(_SAFE)}"] = number
        if not cleaned:
            return
        self._metrics.update(cleaned)
        _guard("log_metrics", lambda: _mlflow().log_metrics(cleaned))

    def log_tags(self, tags: Mapping[str, Any]) -> None:
        if not self.active:
            return
        cleaned = {str(k): _stringify(v) for k, v in tags.items() if v is not None}
        _guard("set_tags", lambda: _mlflow().set_tags(cleaned))

    def log_table(self, rows: list[Mapping[str, Any]], filename: str) -> None:
        """Log a list of records as a queryable artifact.

        Per-ticker and per-strategy detail goes here: metrics alone flatten a table
        into one scalar per cell, which makes comparing ten strategies awkward.
        """
        if not self.active or not rows:
            return
        columns = list(rows[0].keys())
        table = {column: [row.get(column) for row in rows] for column in columns}
        _guard("log_table", lambda: _mlflow().log_table(table, filename))

    @property
    def logged_metrics(self) -> dict[str, float]:
        """What was actually recorded, for assertions and for the response body."""
        return dict(self._metrics)


_DISABLED = Run(active=False)


@contextlib.contextmanager
def track(
    run_name: str,
    tags: Mapping[str, Any] | None = None,
) -> Iterator[Run]:
    """Start a tracked run, or yield an inert handle when tracking is off."""
    settings = get_settings()
    if not settings.mlflow_enabled:
        yield _DISABLED
        return

    try:
        mlflow = _mlflow()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        started = mlflow.start_run(run_name=run_name)
    except Exception:
        logger.warning(
            "could not start an MLflow run at %s; continuing untracked",
            settings.mlflow_tracking_uri,
            exc_info=True,
        )
        yield _DISABLED
        return

    run = Run(active=True, run_id=started.info.run_id)
    run.log_tags({**environment_tags(), **(tags or {})})

    try:
        yield run
    except Exception:
        _guard("end_run(FAILED)", lambda: _mlflow().end_run(status="FAILED"))
        raise
    else:
        _guard("end_run", lambda: _mlflow().end_run(status="FINISHED"))


def environment_tags() -> dict[str, str]:
    """What is needed to reproduce a run beyond its parameters."""
    from app import __version__

    tags = {
        "app_version": __version__,
        "logged_at": dt.datetime.now(dt.UTC).isoformat(),
        "market_data_source": get_settings().market_data_source,
    }
    commit = git_commit()
    if commit:
        tags["git_commit"] = commit
    return tags


def git_commit() -> str | None:
    """Short commit hash, or None outside a repository.

    Without this a logged run records what was measured but not the code that
    measured it, which is half a reproduction recipe.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def data_vintage_tags(tickers: list[str]) -> dict[str, str]:
    """When the underlying prices were ingested, if they came from the store.

    Two runs over the same window can disagree simply because the provider restated
    history between them. Recording the vintage is what makes that explainable
    instead of mysterious.
    """
    settings = get_settings()
    if settings.market_data_source != "parquet":
        return {}

    try:
        from app.services.market_data import get_store

        store = get_store()
        if not store.exists():
            return {}
        coverage = store.coverage()
        if coverage.is_empty():
            return {}
        wanted = {ticker.upper() for ticker in tickers}
        rows = coverage.filter(coverage["ticker"].is_in(list(wanted)))
        if rows.is_empty():
            return {}
        return {
            "data_vintage": str(rows["last_ingested"].max()),
            "data_first_date": str(rows["first_date"].min()),
            "data_last_date": str(rows["last_date"].max()),
        }
    except Exception:
        logger.debug("could not determine the data vintage", exc_info=True)
        return {}


def aggregate(values: list[float | None], name: str) -> dict[str, float]:
    """Mean, min and max of a per-ticker metric, skipping missing values."""
    numbers = [v for v in (_as_float(value) for value in values) if v is not None]
    if not numbers:
        return {}
    return {
        f"{name}_mean": sum(numbers) / len(numbers),
        f"{name}_min": min(numbers),
        f"{name}_max": max(numbers),
    }


# ---------------------------------------------------------------- internals


def _mlflow():
    import mlflow

    return mlflow


def _guard(what: str, action) -> None:
    """Run a tracking call, swallowing failures with a warning."""
    try:
        action()
    except Exception:
        logger.warning("MLflow %s failed; continuing untracked", what, exc_info=True)


def _stringify(value: Any) -> str:
    if isinstance(value, list | tuple):
        return ",".join(str(item) for item in value)
    return str(value)


def _as_float(value: Any) -> float | None:
    import math

    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
