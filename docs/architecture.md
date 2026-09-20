# Architecture

How the pieces fit: the package layout, the HTTP surface, the job seam,
the data pipeline and the browser client.

[← Back to the README](../README.md)

---

## Layout

Everything importable lives under `src/frontier`, so the tests import the
installed package rather than whatever happens to be in the working directory —
a module missing from the wheel fails the suite instead of passing locally and
breaking in Docker.

```
src/frontier/
  api/                the HTTP surface
    main.py           app factory, exception handlers, router wiring
    routers/          one module per resource; thin, no business logic
    schemas/          Pydantic request/response contracts
  services/           adapters between the wire contracts and the core
    tracking.py       MLflow adapter; inert when disabled, never fails a run
  jobs/               job execution
    base.py           JobStore protocol, JobRecord, TaskRef
    memory.py         thread-pool store
    redis_store.py    RQ + Redis store
    progress.py       backend-agnostic progress reporting
  tasks.py            functions workers execute, by dotted path
  cli.py              the `frontier` command
  worker.py           the `frontier-worker` command
  settings.py         env-overridable config
  errors.py           domain exceptions -> HTTP

  data/               price retrieval, returns, features, synthetic source
    loader.py         yfinance downloads, adjusted closes and OHLCV
    features_polars.py  22 features for the whole universe in one Polars pass
    frames.py         the only sanctioned Polars/pandas crossings
    store.py          Parquet price store, DuckDB queries, revision tracking
  forecasting/        forecasting models and their metrics
    torch_lstm.py     PyTorch LSTM
    gbm.py            LightGBM
    datasets.py       leak-free supervised splits
    scaling.py        scalers fitted on training data only
    results.py        SeriesForecast
    train_lstm.py     legacy Keras path
  optimization/       ten strategies + dispatch
    registry.py       the strategy registry and its single dispatch point
    convex.py         cvxpy: min-var, max-Sharpe, risk parity, frontier
    riskfolio_strategies.py  HRP, Gerber, CVaR, CDaR
    constraints.py    box, leverage, group and turnover limits
  portfolio/          construction, performance, selection
    backtest.py       walk-forward engine with costs
  utils/              config, logging, filesystem helpers

frontend/             React + TypeScript, Vite, ECharts
  src/api/            generated schema.ts, fetch client, Frame helpers
  src/charts/         ECharts wrapper, palette bridge, option builders
  src/pages/          one file per page
  src/components/     table, cards, chips, job progress
  nginx.conf          serves the bundle, proxies /api to the service
  Dockerfile          its own build context; the bundle never enters the API image

tests/                321 Python tests, no network access
                      frontend/src/**/*.test.ts: 21 more, in jsdom
scripts/              gen_api_types.py: OpenAPI -> TypeScript
docs/                 this document and its siblings
deploy/
  Dockerfile          one image, two roles (API and worker)
  compose.yaml        UI + API + worker + Redis

pyproject.toml        dependencies, extras, entry points, pytest, ruff, coverage
uv.lock               the resolved set, committed -- 211 packages
.github/workflows/ci.yml   lint, tests, image builds, browser end-to-end
.pre-commit-config.yaml    ruff + biome on staged files (opt in per clone)
```

Dependencies point one way: `api` → `services` → the numerical core
(`data`, `forecasting`, `optimization`, `portfolio`) → `utils`, and the front
end talks only to the HTTP API. Nothing in the numerical core imports from
`api`, so the CLI, the API and the browser all share one implementation and
cannot drift apart.

`settings.py` and `errors.py` sit at the package root rather than inside `api/`
precisely because of that rule: the services read settings and raise these
exceptions, and a service reaching up into the HTTP layer for either would
invert the arrow.

The four numerical packages were once top-level directories named
`Layer1_Preprocessing`, `Layer1_LSTM`, `Layer2_Optimization` and
`Layer3_Portfolio_Generation`, after the pipeline stages of the original
notebook. The names are gone; the boundaries they marked are unchanged.

---

## API surface

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness, and which forecast backends this process has |
| `GET` | `/api/v1/meta/universe` | Default tickers and pipeline parameters |
| `GET` | `/api/v1/meta/strategies` | Strategy catalogue with descriptions |
| `POST` | `/api/v1/data/ingest` | Download OHLCV into the store → `202` + job id |
| `GET` | `/api/v1/data/coverage` | What the store holds, and where it has gaps |
| `POST` | `/api/v1/data/query` | Read-only SQL against the store |
| `POST` | `/api/v1/market/prices` | Adjusted closing prices |
| `POST` | `/api/v1/market/returns` | Simple daily returns |
| `GET` | `/api/v1/meta/backends` | Forecasting backends, scopes and availability |
| `POST` | `/api/v1/forecast` | Forecast prices, derive predicted returns |
| `POST` | `/api/v1/portfolio/optimize` | Weights for each strategy |
| `POST` | `/api/v1/portfolio/performance` | Risk/return statistics per strategy |
| `POST` | `/api/v1/backtest/runs` | Submit a walk-forward backtest → `202` + job id |
| `GET` | `/api/v1/tracking/status` | Whether tracking is on, and where runs live |
| `GET` | `/api/v1/tracking/runs` | Recent tracked runs with params and metrics |
| `POST` | `/api/v1/pipeline/runs` | Submit a full run → `202` + job id |
| `GET` | `/api/v1/pipeline/runs/{id}` | Poll progress (excludes the result) |
| `GET` | `/api/v1/pipeline/runs/{id}/result` | Fetch a completed run |
| `DELETE` | `/api/v1/pipeline/runs/{id}` | Request cancellation |

Tabular payloads use a column-oriented `Frame`:

```json
{ "index": ["2023-01-03", "2023-01-04"],
  "columns": ["AAPL", "MSFT"],
  "data": [[0.0103, -0.0437], [-0.0106, -0.0296]] }
```

This avoids repeating column names on every row and maps directly onto both
`pandas.DataFrame` and the ECharts/Plotly dataset format Phase 9 will use. NaN
and Infinity are serialised as `null`, since neither is valid JSON.


---

## Jobs

`keras_lstm` trains one model per ticker and takes minutes per ticker on CPU, so
the pipeline never runs inside a request. `POST /pipeline/runs` returns `202
Accepted` with a job id; poll `status_url` for progress and read `result_url`
when the state is `succeeded`.

Two interchangeable backends sit behind one protocol:

| | `memory` | `redis` |
|---|---|---|
| Runs work in | the API process (threads) | separate worker processes |
| Survives an API restart | no | **yes** |
| Scales across machines | no | **yes** |
| Can stop a *running* job | no, only flag it | **yes** |
| Needs a broker | no | Redis |

Select with `SO_JOB_BACKEND`:

- **`auto`** (default) — try Redis, fall back to memory if unreachable. Keeps
  `uv run pytest` and `uv run frontier` working on a bare checkout. The fallback
  is logged as a warning, never silent.
- **`redis`** — require Redis; refuse to start without it. **Use this in
  production**, where quietly degrading to a single-process queue is worse than
  not booting.
- **`memory`** — never use Redis.

`GET /health` reports the backend actually in use.

```bash
SO_JOB_BACKEND=redis SO_REDIS_URL=redis://localhost:6379/0 \
  uv run uvicorn frontier.api.main:app

uv run frontier-worker --queues pipeline    # one per spare core
uv run frontier-worker --burst              # drain and exit (CI)
```

### Why jobs are named, not passed

RQ executes work in a **different process**, so a job cannot be a closure — there
is nothing to pickle that would mean anything on the other side. Work is instead
named by a `TaskRef`: an importable dotted path plus JSON-serialisable kwargs.

```python
store.submit(TaskRef("frontier.tasks.run_pipeline", {"request": request.model_dump(mode="json")}))
```

The worker imports `frontier.tasks.run_pipeline` itself. Both backends resolve a
`TaskRef` identically, which is what keeps the in-process store a faithful
stand-in rather than a divergent code path.

Task functions live in `src/frontier/tasks.py` and report progress through
`frontier.jobs.progress.report()`, which writes to the in-memory record or to the RQ
job's `meta` in Redis depending on who is running it — the task does not need to
know.


---

## Polars data pipeline

Feature engineering is Polars: the whole universe is transformed in a single pass with
window expressions over `ticker`, rather than looping tickers in Python and rebuilding
the same 22 rolling columns per series.

2000 rows per ticker, 22 features:

| Tickers | pandas, per-ticker loop | Polars, one pass | Speedup |
|---:|---:|---:|---:|
| 10 | 0.035 s | 0.012 s | 3.0× |
| 200 | 0.601 s | 0.212 s | 2.8× |
| 1000 | 2.791 s | 1.223 s | 2.3× |

**About 2.3–3.1×, not the "5–30×" this migration was pitched on.** That estimate was
wrong for this workload: pandas' rolling operations are already compiled C, so there
is no interpreted inner loop to remove. The gain comes from parallelism across tickers
and from not paying per-series overhead 22 times.

### The speedup exposed the real bottleneck

With features 3× faster, sequence construction for the LSTM became the dominant cost —
41.5 s for 500 tickers at a 60-step window, because every window was stacked into a new
array. Switching to `sliding_window_view` and returning the **view** rather than
materialising it took that to **2.9 s**, a 14× win and a bigger improvement than Polars
itself delivered.

### Where the boundary sits

Polars from the data source up to the point where the numbers become a matrix; numpy
and pandas from there on. cvxpy and the models want numpy, and Riskfolio-Lib's API
requires pandas. `src/frontier/data/frames.py` holds the only sanctioned crossings.

Polars having no index is an advantage beyond speed: several of the original defects
here were index-alignment hazards — a frame silently reindexed, a `droplevel` that
removed the wrong level, columns assigned positionally.


---

## Price store

**yfinance restates history.** Splits get re-adjusted and bad ticks corrected, so the
adjusted close for a date five years ago is not guaranteed to be the number it returned
yesterday. A pipeline that re-downloads on every run cannot reproduce its own results.

```bash
curl -X POST localhost:8000/api/v1/data/ingest \
  -d '{"tickers":["AAPL","MSFT"],"start":"2010-01-01","end":"2024-01-01"}'

SO_MARKET_DATA_SOURCE=parquet uv run uvicorn frontier.api.main:app
```

Prices are ingested once into Parquet, partitioned by ticker, each row stamped with the
`ingested_at` of the batch that wrote it. Re-ingesting is idempotent; a **restated** bar
is reported rather than overwritten silently:

```json
{ "rows_added": 0, "rows_revised": 23, "revised_tickers": ["AAPL"],
  "revisions": [{"date":"2020-01-02","old_adj_close":166.69,"new_adj_close":83.35}] }
```

The `parquet` source never falls back to the network — an unstored ticker is a `502`
telling you to ingest it, because a quiet download would undo the whole point.

`GET /data/coverage` reports rows, spans, vintages and **gaps**, which is how an
interrupted ingestion or a delisting surfaces before a backtest silently spans it.

### Why Parquet, and why both engines

1,565,400 rows × 8 columns:

| Format | Size | Read |
|---|---:|---:|
| CSV | 208.3 MB | 49 ms |
| Parquet + zstd | **59.3 MB** | **27 ms** |

| Operation | Polars | DuckDB |
|---|---:|---:|
| Filtered bulk read (26,080 rows) | **9.1 ms** | 36.9 ms |
| `GROUP BY ticker` over 1.57M rows | 38.3 ms | **36.3 ms** |

**DuckDB is not faster than Polars at anything measured here.** Polars is 4× quicker on
the read that feeds the pipeline. DuckDB earns its place on *expressiveness*: gap
detection is a `LAG` over a partition, and `POST /data/query` lets someone interrogate
the store without writing Python. That endpoint accepts a single `SELECT`/`WITH` and
nothing else, conservatively enough that `SELECT 'create'` is refused — DuckDB can
`COPY` to the filesystem and `ATTACH` databases.


---

## Front end

React 19 + TypeScript on Vite, charts in ECharts, six pages against the same API
the CLI uses. `npm run dev` proxies to the service;
`docker compose -f deploy/compose.yaml up` serves the built bundle from nginx on
<http://127.0.0.1:8080>.

| Page | What it does |
|---|---|
| **Run** | Submits a pipeline run, polls it with a progress bar, and renders forecast quality, weights, growth and performance when it lands |
| **Forecast** | One forecast in detail: predicted against actual per ticker, MASE against the random-walk baseline |
| **Portfolio** | Ten strategies on realised returns, box constraints, and the efficient frontier with the selected strategy marked |
| **Backtest** | Walk-forward run as a job: net against gross growth, cost drag, and the out-of-sample table |
| **Data** | Store coverage and gaps, ingest as a job, and read-only SQL |
| **Tracking** | MLflow runs with their parameters, metrics, git commit and data vintage |

### Types are generated, not copied

`scripts/gen_api_types.py` reads the OpenAPI schema straight off the app and
writes `frontend/src/api/schema.ts`. A renamed response field therefore fails
`tsc`, not a browser. Re-generate after changing anything in `src/frontier/api/schemas/`:

```bash
uv run python scripts/gen_api_types.py
```

It already caught two things. `PipelineResult.warnings` and `.failed` have
defaults server-side but are *not* required in the schema, so the pages that
rendered them unconditionally would have thrown on a response that omitted them.
And the result endpoint published no schema at all for what a finished job
contains — `JobStatus.result` was typed `Any`, since the job store is generic
and keeps whatever a task returns. `src/frontier/api/schemas/job_results.py` now names the
three shapes a task actually produces, so `GET /pipeline/runs/{id}/result`
documents them and the front end generates real types instead of `unknown`.

### Charts

The palette is a validated eight-hue categorical ramp, declared once in
`styles.css` as custom properties and read back by `charts/theme.ts`, so an
option object can never disagree with the page around it. Both modes clear the
colour-vision-deficiency separation floor; dark is its own set of steps against
the dark surface, not an inverted light palette.

Three rules the code enforces rather than documents:

- **Hues are never cycled.** There are ten strategies and eight slots. Indexing
  colour off the catalogue position — the obvious implementation — gave
  `Markowitz_MaxSharpe` and `MinCVaR` the same blue, which the first screenshot
  caught. Slots are allocated to what is on the chart instead, and capped at
  eight; past that the UI folds or facets rather than inventing a ninth hue.
- **Filtering never repaints the survivors.** An allocated slot stays with its
  strategy until that strategy leaves the chart, so toggling one line does not
  recolour the rest. `makeSlots` in `charts/theme.ts`, with tests.
- **Every chart has a table beside it.** Three light-mode hues fall below 3:1
  contrast against the surface; the rule for that is relief — a readable table
  view or direct labels — and each chart here ships one or both.

Charts are also where the browser cost is: ECharts is 596 kB of the bundle and
sits in its own chunk, so an app edit does not invalidate it in a cache.

### What the tests cover

`npm test` runs 21 tests in jsdom: the Frame-to-series transforms, the
formatters, the colour-slot allocator, and three mounting tests that boot the
app against a stubbed API — one of which drives a pipeline job to a rendered
result. They are not screenshots; jsdom paints nothing. What they catch is the
class of failure that actually happens: a null field reaching a table, a bad
hook, a page that renders nothing when the API is down.
