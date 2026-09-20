# Frontier

**LSTM forecasting & portfolio optimization.**

Forecast equity prices, optimize portfolios across ten strategies, and compare
their risk-adjusted performance.

FastAPI + Pydantic service layer, RQ + Redis job queue, PyTorch + LightGBM
forecasting, cvxpy + Riskfolio-Lib optimizers, Polars data pipeline, Parquet +
DuckDB price store, MLflow tracking, uv + Docker packaging, a React + TypeScript
front end with ECharts, and CI that lints, tests, builds both images and drives
the whole stack in a browser.

One implementation, three ways in: a CLI, an HTTP API, and a browser.

---

## Quick start

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.12 itself, so
nothing else is a prerequisite.

```bash
uv sync                  # .venv with runtime + dev dependencies, from uv.lock
```

Run the API:

```bash
uv run uvicorn frontier.api.main:app --reload
```

Interactive docs at <http://127.0.0.1:8000/docs>.

Run a worker (needs Redis — see [Jobs](docs/architecture.md#jobs)):

```bash
docker run -d -p 6379:6379 redis:7-alpine     # or: sudo pacman -S redis && sudo systemctl start redis
uv run frontier-worker
```

Without Redis the API still works: it falls back to an in-process queue and says
so in `GET /health`.

Run the CLI:

```bash
uv run frontier --tickers AAPL MSFT JPM --backend naive
```

Run the tests:

```bash
uv run pytest                    # 321 tests
uv run ruff check . && uv run ruff format --check .
```

Run the front end (needs the API above on port 8000):

```bash
cd frontend && npm install && npm run dev      # http://localhost:5173
```

Or bring up the whole stack — UI, API, worker and Redis:

```bash
docker compose -f deploy/compose.yaml up --build     # http://127.0.0.1:8080
```

`export COMPOSE_FILE=deploy/compose.yaml` drops the flag for the rest of the
shell; CI sets it the same way.

`uv run <cmd>` syncs the environment first, so it is always in step with
`uv.lock`. `.venv/bin/<cmd>` works too once `uv sync` has run. The legacy
TensorFlow backend is optional and left out by default (`--extra keras`); see
[Packaging](docs/operations.md#packaging).

---

## Repository layout

```
src/frontier/         the installable package -- one import name, one wheel
  api/                HTTP surface: routers, Pydantic contracts, app factory
  services/           adapters between the wire contracts and the core
  jobs/               the async job seam: in-process queue, or RQ + Redis
  data/               market data, Parquet store, returns, features
  forecasting/        forecasting models and the metrics that judge them
  optimization/       ten optimizers and the strategy registry
  portfolio/          construction, walk-forward backtest, performance
  utils/              configuration constants, logging, filesystem helpers
  cli.py  tasks.py  worker.py  settings.py  errors.py

frontend/             React + TypeScript, Vite, ECharts
tests/                321 Python tests, no network access
scripts/              gen_api_types.py: OpenAPI -> TypeScript
deploy/               Dockerfile (API + worker) and compose.yaml
docs/                 the long-form documentation indexed below
```

Dependencies point one way: `api` → `services` → the numerical core →
`utils`, and the front end talks only to the HTTP API. Nothing in the numerical
core imports from `api`, so the CLI, the API and the browser all share one
implementation and cannot drift apart.

The four packages under `src/frontier` that carry the mathematics — `data`,
`forecasting`, `optimization`, `portfolio` — were once top-level `Layer1_*`,
`Layer2_*` and `Layer3_*` directories named after pipeline stages. The names
went; the boundaries did not.

---

## Documentation

| Document | What is in it |
|---|---|
| [Methodology](docs/methodology.md) | What each number means and how much work it is doing: why prices are not returns, why "93% accuracy" was a property of the metric, every backend measured on real data, the strategies, the walk-forward backtest |
| [Architecture](docs/architecture.md) | The package layout, the API surface, the job seam, the Polars pipeline, the Parquet price store, the front end |
| [Operations](docs/operations.md) | Packaging with uv and Docker, MLflow experiment tracking, and the quality gates CI enforces |
| [Roadmap and limitations](docs/roadmap.md) | What this deliberately does not do, and the ten migration phases that got it here |

**Start with [Methodology](docs/methodology.md)** if you intend to quote a
result from this project. The headline accuracy figure of the original notebook
was an artefact of predicting price levels from price levels; the document
explains what replaced it and what the honest numbers look like.
