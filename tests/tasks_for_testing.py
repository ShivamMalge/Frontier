"""Task functions used by the job-store tests.

These live at module level, with a real importable dotted path, because that is
exactly the constraint RQ imposes: a job must be resolvable by name in a worker
process that never saw the code that enqueued it. Testing through the same
interface keeps the in-process store honest as a stand-in for the Redis one.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from app.jobs.progress import report

#: Dotted paths, so tests need not retype them.
ECHO = "tests.tasks_for_testing.echo"
BLOCK_UNTIL_RELEASED = "tests.tasks_for_testing.block_until_released"
FAIL = "tests.tasks_for_testing.fail"
COUNT_UP = "tests.tasks_for_testing.count_up"


def echo(value: str = "ok") -> dict[str, str]:
    return {"echoed": value, "pid": str(os.getpid())}


def count_up(steps: int = 4) -> list[float]:
    """Reports progress, so the reporting path is covered for both backends."""
    seen = []
    for step in range(1, steps + 1):
        fraction = step / steps
        report(fraction, f"step {step} of {steps}")
        seen.append(fraction)
    return seen


def fail(message: str = "synthetic task failure") -> None:
    raise RuntimeError(message)


def signal_path(token: str) -> Path:
    """A file used to coordinate with a task that may run in another process."""
    return Path(tempfile.gettempdir()) / f"frontier-test-{token}"


def block_until_released(token: str, timeout: float = 30.0) -> str:
    """Occupy a worker until its signal file appears.

    Filesystem-based rather than in-memory so it works whether the task runs in
    this process (in-memory store) or another one (RQ).
    """
    marker = signal_path(token)
    started = signal_path(f"{token}-started")
    started.touch()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists():
            return "released"
        time.sleep(0.01)
    return "timed out"
