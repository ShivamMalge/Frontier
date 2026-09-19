# main.py

"""Command-line entry point.

A thin wrapper over the same service layer the API uses, so the CLI and the HTTP
surface can never drift apart. Run ``python main.py --help`` for options.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from app.schemas.pipeline import PipelineRequest
from app.services import pipeline as pipeline_service
from app.services.forecasting import available_backends
from app.settings import get_settings
from utils.helpers import ensure_dirs
from utils.logger import log


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="frontier",
        description="Forecast prices, optimize portfolios and compare strategies.",
    )
    parser.add_argument("--tickers", nargs="+", default=settings.universe)
    parser.add_argument("--start", default=settings.start_date)
    parser.add_argument("--end", default=settings.end_date)
    parser.add_argument(
        "--backend",
        default=settings.default_forecast_backend,
        choices=available_backends(),
        help="'naive' is the random-walk baseline; 'keras_lstm' trains one model per ticker.",
    )
    parser.add_argument("--risk-tolerance", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed"),
        help="Directory for the CSV and JSON outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_dirs()

    request = PipelineRequest(
        tickers=args.tickers,
        start=dt.date.fromisoformat(args.start),
        end=dt.date.fromisoformat(args.end),
        backend=args.backend,
        risk_tolerance=args.risk_tolerance,
        epochs=args.epochs,
    )

    def report(fraction: float, message: str) -> None:
        log(f"[{fraction:5.1%}] {message}")

    result = pipeline_service.run(request, report)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pipeline_result.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2)
    )

    log("")
    log(f"Backend: {result.backend}")
    log("Forecast quality (MASE below 1.0 beats a random walk):")
    for metric in result.forecast_metrics:
        log(
            f"  {metric.ticker:6s} MASE={metric.mase_vs_naive:6.3f}  "
            f"dir={metric.directional_accuracy:5.1%}  RMSE={metric.rmse:8.3f}"
        )

    log("")
    log("Strategy performance:")
    log(f"  {'strategy':22s} {'return':>8s} {'vol':>8s} {'sharpe':>8s} {'maxDD':>8s}")
    for row in result.performance:
        log(
            f"  {row.strategy:22s} {row.annual_return:8.2%} {row.annual_volatility:8.2%} "
            f"{row.sharpe:8.3f} {row.max_drawdown:8.2%}"
        )

    log("")
    log(f"Selected for risk tolerance {args.risk_tolerance:.2f}: {result.selected_strategy}")
    for ticker, weight in sorted(
        result.selected_weights.items(), key=lambda kv: -kv[1]
    ):
        if weight > 1e-4:
            log(f"  {ticker:6s} {weight:7.2%}")

    for warning in result.warnings:
        log(f"WARNING {warning}")
    for failure in result.failed:
        log(f"SKIPPED {failure.ticker}: {failure.reason}")

    log("")
    log(f"Wrote {args.out / 'pipeline_result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
