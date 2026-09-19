"""RQ worker entry point.

    frontier-worker                         # one worker on the default queue
    python -m app.worker                    # the same thing, without an install
    python -m app.worker --queues pipeline  # explicit queue
    python -m app.worker --burst            # drain the queue, then exit (for CI)

Run as many of these as there are cores to spare. Because forecasting is
CPU-bound, workers scale independently of the web tier -- which is the main
reason for moving off the in-process executor.
"""

from __future__ import annotations

import argparse
import logging
import sys

from redis import Redis
from rq import Worker

from app.settings import get_settings

logger = logging.getLogger("app.worker")


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="frontier-worker", description="Run an RQ worker.")
    parser.add_argument("--queues", nargs="+", default=[settings.queue_name])
    parser.add_argument("--url", default=settings.redis_url)
    parser.add_argument(
        "--burst",
        action="store_true",
        help="Process what is queued, then exit instead of waiting for more.",
    )
    parser.add_argument("--name", default=None, help="Worker name; defaults to host.pid.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)

    try:
        connection = Redis.from_url(args.url)
        connection.ping()
    except Exception as exc:  # noqa: BLE001 -- any failure here means the same thing: no broker
        logger.error("cannot reach Redis at %s: %s", args.url, exc)
        logger.error("start one with: docker run -p 6379:6379 redis:7-alpine")
        return 1

    logger.info("worker listening on %s (redis %s)", ", ".join(args.queues), args.url)
    worker = Worker(args.queues, connection=connection, name=args.name)
    worker.work(burst=args.burst, with_scheduler=False)
    return 0


def run() -> None:
    """Console-script entry point (``frontier-worker``)."""
    sys.exit(main())


if __name__ == "__main__":
    run()
