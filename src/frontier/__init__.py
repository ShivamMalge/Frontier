"""Frontier: LSTM forecasting and portfolio optimization, behind a FastAPI service.

The package is layered, and the dependency arrows only ever point downwards:

    api/            HTTP surface -- routers, Pydantic contracts, the app factory
    cli.py          the same pipeline at a terminal, without the HTTP hop
    services/       adapters between the wire contracts and the numerical core
    jobs/           the async job seam: in-process queue, or RQ + Redis
    tasks.py        the units of work a worker imports and runs
    worker.py       the RQ worker entrypoint

The numerical core knows nothing about HTTP, jobs or Pydantic:

    data/           market data, the Parquet store, returns and features
    forecasting/    forecasting models and the metrics that judge them
    optimization/   portfolio optimizers and the strategy registry
    portfolio/      portfolio construction, backtesting and performance

Cross-cutting, used by every layer:

    settings.py     one Settings object, read from the environment
    errors.py       the exception types the API translates into responses
    utils/          configuration constants, logging, filesystem helpers
"""

__version__ = "0.1.0"
