# Frontier

**LSTM forecasting & portfolio optimization.**

Forecast equity prices, optimize portfolios across six strategies, and compare
their risk-adjusted performance.

**Migration complete: all 10 phases**, plus walk-forward backtesting — FastAPI +
Pydantic service layer, RQ + Redis job queue, PyTorch + LightGBM forecasting,
cvxpy + Riskfolio-Lib optimizers, Polars data pipeline, Parquet + DuckDB price
store, MLflow tracking, uv + Docker packaging, a React + TypeScript front end
with ECharts, and CI that lints, tests, builds both images and drives the whole
stack in a browser. See [Migration roadmap](#migration-roadmap).

---

## Quick start

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.12 itself, so
nothing else is a prerequisite.

```bash
uv sync                  # .venv with runtime + dev dependencies, from uv.lock
```

Run the API:

```bash
uv run uvicorn app.main:app --reload
```

Interactive docs at <http://127.0.0.1:8000/docs>.

Run a worker (needs Redis — see [Jobs](#jobs)):

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
docker compose up --build                      # http://127.0.0.1:8080
```

`uv run <cmd>` syncs the environment first, so it is always in step with
`uv.lock`. `.venv/bin/<cmd>` works too once `uv sync` has run. The legacy
TensorFlow backend is optional and left out by default (`--extra keras`); see
[Packaging](#packaging).

---

## Read this before trusting a number

Two things in this project are easy to misread. Both are now surfaced in the API
rather than hidden.

### Prices are not returns

The forecasting models emit **price levels**. The optimizers consume **returns**.
`POST /api/v1/forecast` returns both; `predicted_returns` is the field to feed
into `POST /api/v1/portfolio/optimize`.

Earlier versions of this pipeline passed predicted *prices* straight into the
optimizers as though they were returns. Annualising a $150 price level as a daily
return produces an expected annual return of 37,800 and a cumulative growth curve
that overflows. Every weight and every performance figure was meaningless as a
result. `POST /portfolio/optimize` now rejects payloads whose mean absolute value
exceeds 1.0, so the mistake cannot recur silently.

### "93% accuracy" was a property of the metric, not the model

The original accuracy figure came from `100 * (1 - RMSE / mean(price))`. That is
not accuracy in any forecasting sense — it is scale-dependent, so the same model
scores higher on more expensive stocks, and a do-nothing random walk scores above
97 on it.

Judge forecasts on **`mase_vs_naive`** (below 1.0 beats doing nothing; it is the
model's mean absolute error divided by a random walk's) and
**`directional_accuracy`** (0.5 is chance). The legacy figure is still returned as
`legacy_approximate_accuracy`, labelled for what it is.

### Every backend, measured on real data

AAPL / MSFT / JPM / GS / PFE / LMT, 2015-01-02 → 2022-12-30, 2014 trading days,
2/3 train split, single CPU. Averaged across the six tickers:

| Backend | Configuration | Seconds | MASE | Directional | R² | Legacy "accuracy" |
|---|---|---:|---:|---:|---:|---:|
| `naive` | random walk baseline | 0.0 | **1.000** | n/a | +0.989 | 98.21 |
| `keras_lstm` | *the original*: univariate price, per ticker | 56.3 | **2.928** | 0.501 | +0.912 | **95.04** |
| `torch_lstm` | price target, per ticker | 13.2 | 35.674 | 0.492 | −9.996 | 48.97 |
| `torch_lstm` | return target, per ticker | 16.6 | **1.011** | 0.518 | +0.988 | 98.19 |
| `torch_lstm` | return target, one shared model | 23.5 | 1.031 | 0.512 | +0.987 | 98.16 |
| `lightgbm` | return target, per ticker | **0.4** | **1.011** | 0.510 | +0.988 | 98.21 |
| `lightgbm` | return target, one shared model | 0.5 | 1.040 | 0.503 | +0.987 | 98.16 |

Four things to read off that table.

**The original model was about three times worse than doing nothing.** MASE 2.93
against a naive baseline of 1.00 — while scoring 95.04 on the legacy metric, which
reproduces the remembered "93%+ across all stocks". The metric was the result, not
the model.

**Fixing the formulation cut forecast error by 2.9×**, from MASE 2.93 to 1.011.
What changed: a return target instead of a price level, 22 engineered features
instead of one raw price, and a scaler fitted on training data only.

**Nothing beats the random walk.** MASE 1.011 essentially *matches* the baseline,
and directional accuracy of ~0.51 across every backend is a coin flip. That is the
honest result for one-day-ahead equity price forecasting, and it is the finding —
not a defect in the implementation. Anything here claiming to beat 1.000 by a wide
margin would mean a leak, which is why the tests assert MASE stays above 0.9 on
synthetic random-walk data.

**LightGBM gets the same accuracy 140× faster** — 0.4 s against 56.3 s. That is
what makes the pipeline usable interactively, and it is why the original had to be
hardcoded for the demo.

### Why the price target collapses

The `torch_lstm` price-target row (MASE 35.7, R² −10.0) is the leak made visible.
Predicting tomorrow's *price level* requires extrapolating beyond the training
range, and with the scaler fitted only on training data the model cannot. The
original scored well on this target precisely *because* its `MinMaxScaler` was
fitted on the full series before splitting, so the test window's high and low were
baked into the normalisation. Remove that and price-level modelling falls apart.

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
  uv run uvicorn app.main:app

uv run frontier-worker --queues pipeline    # one per spare core
uv run frontier-worker --burst              # drain and exit (CI)
```

### Why jobs are named, not passed

RQ executes work in a **different process**, so a job cannot be a closure — there
is nothing to pickle that would mean anything on the other side. Work is instead
named by a `TaskRef`: an importable dotted path plus JSON-serialisable kwargs.

```python
store.submit(TaskRef("app.tasks.run_pipeline", {"request": request.model_dump(mode="json")}))
```

The worker imports `app.tasks.run_pipeline` itself. Both backends resolve a
`TaskRef` identically, which is what keeps the in-process store a faithful
stand-in rather than a divergent code path.

Task functions live in `app/tasks.py` and report progress through
`app.jobs.progress.report()`, which writes to the in-memory record or to the RQ
job's `meta` in Redis depending on who is running it — the task does not need to
know.

---

## Forecasting

| Backend | Scope | Target | Needs | Notes |
|---|---|---|---|---|
| `naive` | per ticker | price | — | Random walk. MASE 1.0 by definition. |
| `lightgbm` | universe | return | `lightgbm` | Fastest and as accurate as anything here. Reports feature importances. |
| `torch_lstm` | universe | return | `torch` | Two LSTM layers, ~129k params, multivariate. |
| `keras_lstm` | per ticker | price | `tensorflow` | Legacy. Kept to reproduce the original result. |

`GET /api/v1/meta/backends` lists what this process actually has; a missing
library makes its backend unavailable rather than failing on first use.

**Target.** `return` (default) predicts tomorrow's simple return and rebuilds a
price as `last_observed_close * (1 + predicted_return)`. `price` predicts the level
directly, as the original did — kept so the difference can be demonstrated.

**`multi_series`.** Trains one model across the whole universe instead of one per
ticker. One artefact to version and evaluate, and the model can use structure
shared between tickers. It is *not* necessarily faster: pooling six tickers means
one model sees six times the data, so total work is similar. The win is
operational, not wall-clock.

**Features** (`Layer1_Preprocessing/feature_engineering.py`, 22 columns): 10 lagged
returns, realised volatility over 5/21/63 days, momentum over 5/21/63 days, gap to
the 21- and 63-day moving average, position within the 21-day range, RSI(14), and
weekday/month. Every column on row *t* uses only data available at the close of
*t*; the target is the move from *t* to *t+1*. Volume and cross-sectional features
are absent because the loader fetches adjusted closes only — widening that is a
Phase 6 concern.

**Validation.** Both new backends early-stop on the last 15% of the *training*
window, taken chronologically. A random validation split would let the future
inform model selection.

---

## Layout

```
app/                  FastAPI service (Phase 1)
  routers/            HTTP surface, thin
  schemas/            Pydantic contracts
  services/           adapters onto the numerical core
    tracking.py       MLflow adapter; inert when disabled, never fails a run
  jobs/               job execution (Phase 2)
    base.py           JobStore protocol, JobRecord, TaskRef
    memory.py         thread-pool store
    redis_store.py    RQ + Redis store
    progress.py       backend-agnostic progress reporting
  tasks.py            functions workers execute, by dotted path
  cli.py              the `frontier` command
  worker.py           the `frontier-worker` command
  settings.py         env-overridable config
  errors.py           domain exceptions -> HTTP

Layer1_Preprocessing/ price retrieval, returns, features, synthetic source
    features_polars.py  22 features for the whole universe in one Polars pass
    frames.py         the only sanctioned Polars/pandas crossings
    store.py          Parquet price store, DuckDB queries, revision tracking
Layer1_LSTM/          forecasting (name is historical, not LSTM-only)
    torch_lstm.py     PyTorch LSTM
    gbm.py            LightGBM
    datasets.py       leak-free supervised splits
    scaling.py        scalers fitted on training data only
    results.py        SeriesForecast
    train_lstm.py     legacy Keras path
Layer2_Optimization/  ten strategies + dispatch
    convex.py         cvxpy: min-var, max-Sharpe, risk parity, frontier
    riskfolio_strategies.py  HRP, Gerber, CVaR, CDaR
    constraints.py    box, leverage, group and turnover limits
Layer3_Portfolio_Generation/  construction, performance, selection
    backtest.py       walk-forward engine with costs
frontend/             React + TypeScript, Vite, ECharts (Phase 9)
    src/api/          generated schema.ts, fetch client, Frame helpers
    src/charts/       ECharts wrapper, palette bridge, option builders
    src/pages/        one file per page
    src/components/   table, cards, chips, job progress
    nginx.conf        serves the bundle, proxies /api to the service
scripts/              gen_api_types.py: OpenAPI -> TypeScript
utils/                config, logging, filesystem helpers
tests/                321 Python tests, no network access
                      frontend/src/**/*.test.ts: 21 more, in jsdom

pyproject.toml        dependencies, extras, entry points, pytest, ruff, coverage
uv.lock               the resolved set, committed -- 211 packages
Dockerfile            one image, two roles (API and worker)
compose.yaml          UI + API + worker + Redis
.github/workflows/ci.yml   lint, tests, image builds, browser end-to-end
.pre-commit-config.yaml    ruff + biome on staged files (opt in per clone)
main.py               shim so `python main.py` works without an install
```

Dependencies point one way: `app` → `Layer*` → `utils`, and the front end talks
only to the HTTP API. Nothing in the numerical core imports from `app`, so the
CLI, the API and the browser all share one implementation and cannot drift
apart.

---

## Strategies

Ten strategies, solved by cvxpy (convex) or Riskfolio-Lib (hierarchical,
robust-covariance, tail-risk). `GET /api/v1/meta/strategies` returns this table
with descriptions.

| Strategy | Family | Solver | Long-only | Uses forecast | Honours constraints |
|---|---|---|---|---|---|
| `Markowitz_MaxSharpe` | mean-variance | cvxpy | yes | **yes** | yes |
| `Markowitz_MinVar` | mean-variance | cvxpy | yes | no | yes |
| `RiskParity` | risk-based | cvxpy | yes | no | no |
| `GMV` | mean-variance | cvxpy | **no** | no | yes |
| `HRP` | hierarchical | riskfolio | yes | no | no |
| `HRP_CVaR` | hierarchical | riskfolio | yes | no | no |
| `Gerber_InvVar` | robust-covariance | riskfolio | yes | no | no |
| `Gerber_HRP` | robust-covariance | riskfolio | yes | no | no |
| `MinCVaR` | tail-risk | riskfolio | yes | no | yes |
| `MinCDaR` | tail-risk | riskfolio | yes | no | yes |

Worth knowing: **only `Markowitz_MaxSharpe` uses the return forecast.** The other
nine depend solely on the covariance matrix or the return distribution, so a better
forecasting model cannot improve them. `GMV` and `Markowitz_MinVar` are the same
objective — the only difference is that GMV permits shorts.

### Measured on real data

AAPL / MSFT / ADBE / AMD / JPM / GS / PFE / JNJ / LMT / BA, 2018–2022, 1258 trading
days. All ten solved in **2.4 s**. `rc spread` is the gap between the largest and
smallest risk contribution — zero means true risk parity.

| Strategy | Return | Vol | Sharpe | Sortino | Max DD | Gross | Held | rc spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Markowitz_MaxSharpe` | 29.10% | 27.70% | **1.050** | 1.520 | −31.78% | 1.00 | 5 | 0.4580 |
| `Markowitz_MinVar` | 12.55% | 18.99% | 0.661 | 0.931 | −29.11% | 1.00 | 7 | 0.5233 |
| `RiskParity` | 17.95% | 22.56% | 0.796 | 1.117 | −33.39% | 1.00 | 10 | **0.0000** |
| `GMV` | 12.81% | **18.85%** | 0.680 | 0.967 | −27.11% | 1.17 | 10 | 0.5642 |
| `HRP` | 16.00% | 20.49% | 0.781 | 1.101 | −31.91% | 1.00 | 10 | 0.1698 |
| `HRP_CVaR` | 17.57% | 21.74% | 0.808 | 1.138 | −32.70% | 1.00 | 10 | 0.0792 |
| `Gerber_InvVar` | 16.78% | 21.35% | 0.786 | 1.109 | −32.12% | 1.00 | 10 | 0.1103 |
| `Gerber_HRP` | 15.79% | 20.46% | 0.772 | 1.088 | −32.14% | 1.00 | 10 | 0.1752 |
| `MinCVaR` | 12.92% | 19.26% | 0.671 | 0.952 | −29.56% | 1.00 | 5 | 0.3527 |
| `MinCDaR` | 13.94% | 20.27% | 0.688 | 0.983 | **−26.56%** | 1.00 | 4 | 0.6899 |

Three of those numbers are the Phase 4 fixes showing up:

- **`RiskParity` reaches an rc spread of exactly 0.0000.** Before Phase 4 it
  silently returned equal weights, whose risk contributions are nothing like equal.
- **`MinCDaR` has the shallowest drawdown**, which is precisely what conditional
  drawdown-at-risk minimises. No prior strategy targeted the path of losses.
- **`GMV` is the only one with gross exposure above 1.0** (1.17), confirming it is
  the short-permitting twin of `Markowitz_MinVar` and nothing more.

### Constraints

`POST /portfolio/optimize` and `POST /pipeline/runs` accept a `constraints` object.
This is the practical reason for leaving `scipy.optimize`: each of these is a linear
inequality the solver satisfies exactly or declares infeasible, rather than another
hand-written callback that may quietly fail to converge.

| Field | Meaning |
|---|---|
| `min_weight` / `max_weight` | Per-asset box. Negative `min_weight` permits shorts. |
| `max_leverage` | Cap on gross exposure `sum |w|`. 1.3 is a 130/30 mandate. |
| `group_caps` / `group_floors` | Named ceilings and floors, e.g. `{"banks": [["JPM","GS"], 0.2]}`. |
| `max_turnover` + `previous_weights` | Cap on `sum |w − w_prev|`. |

Same universe, `Markowitz_MinVar` only:

| Case | Vol | Max weight | Gross | Banks | Turnover |
|---|---:|---:|---:|---:|---:|
| unconstrained | 18.99% | 52.33% | 1.00 | 5.14% | 1.169 |
| `max_weight` 15% | 21.47% | **15.00%** | 1.00 | 24.50% | 0.523 |
| banks cap 20% | 18.99% | 52.33% | 1.00 | 5.14% | 1.169 |
| 130/30 short mandate | 18.85% | 50.71% | **1.17** | 10.06% | 1.171 |
| `max_turnover` 10% | 23.63% | 15.00% | 1.00 | 20.00% | **0.100** |

Each limit is hit exactly. The banks cap row is unchanged because the unconstrained
portfolio already holds only 5.14% in banks — the cap is non-binding, not ignored.
Note also that constraints cost return: capping weights at 15% raises volatility
from 18.99% to 21.47%, and a 10% turnover budget raises it to 23.63%.

Strategies marked "honours constraints: no" derive every weight from their own
construction, leaving nothing to constrain. Passing constraints to those returns a
**warning naming them** rather than silently ignoring the request.

### Efficient frontier

`POST /api/v1/portfolio/frontier` minimises variance at each of `points` target
returns between the minimum-variance and maximum-return portfolios, so the curve
spans exactly what the constraints allow. `tangency_index` marks the max-Sharpe
point, and a test asserts it agrees with `Markowitz_MaxSharpe` — on the data above,
both give Sharpe 1.050.

The repo has shipped a `plot_effiecient_frontier.py` since the start. It plotted
cumulative growth. This computes the actual curve.

## Walk-forward backtest

`POST /api/v1/backtest/runs` re-optimises on a **trailing** window at each rebalance
and holds forward through data the optimizer never saw. Every other performance
number in this repository is in-sample; this one is not.

Three details decide whether a backtest means anything, and all three are handled:

- **Weights drift.** Between rebalances the portfolio is *held*, so weights move with
  prices (`w' = w(1+r)/(1+R)`). Pinning them to target daily implies free trading and
  quietly inflates returns.
- **Turnover is measured against the drifted weights**, not the previous target. That
  difference is what actually has to be traded.
- **Trading is charged.** `cost_bps` applies to traded notional at each rebalance.

### The result that matters

Ten tickers, 2016–2022, monthly rebalancing on a one-year window at 10 bps. 72
rebalances, 1509 out-of-sample days.

| Strategy | In-sample Sharpe | Walk-forward Sharpe | Gap | Turnover/yr | Cost drag |
|---|---:|---:|---:|---:|---:|
| `Markowitz_MaxSharpe` | **1.251** | 0.814 | **−0.438** | 5.63 | 0.56% |
| `Markowitz_MinVar` | 0.891 | 0.736 | −0.154 | 1.84 | 0.18% |
| `RiskParity` | 1.042 | 0.968 | −0.074 | **0.83** | **0.08%** |
| `GMV` | 0.891 | 0.717 | −0.174 | 2.97 | 0.30% |
| `HRP` | 0.996 | 0.998 | **+0.002** | 2.16 | 0.22% |
| `HRP_CVaR` | 1.028 | **1.004** | −0.024 | 2.58 | 0.26% |
| `Gerber_InvVar` | 1.006 | 0.932 | −0.075 | 0.93 | 0.09% |
| `Gerber_HRP` | 0.987 | 0.989 | **+0.002** | 2.04 | 0.20% |
| `MinCVaR` | 0.859 | 0.822 | −0.038 | 3.64 | 0.36% |
| `MinCDaR` | 0.904 | 0.725 | −0.179 | 4.42 | 0.44% |

**The strategy that looks best in-sample is not the one that survives.**
`Markowitz_MaxSharpe` ranks first in-sample at 1.251 and fifth out of sample at
0.814 — the largest degradation of any strategy. The out-of-sample winner is
`HRP_CVaR` (1.004), and the two HRP variants are the only ones that do not degrade.

That ordering is not a coincidence: **`Markowitz_MaxSharpe` is the only strategy that
uses the return forecast**, so it is the only one exposed to estimation error in
expected returns — and the forecast has no measurable edge. The risk-based methods
ignore expected returns entirely and hold up. It also churns the most: 5.63× turnover
a year against `RiskParity`'s 0.83×.

Eight of ten strategies score worse out of sample than in.

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
requires pandas. `Layer1_Preprocessing/frames.py` holds the only sanctioned crossings.

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

SO_MARKET_DATA_SOURCE=parquet uv run uvicorn app.main:app
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

## Experiment tracking

The original project reported "93%+ accuracy" with no record of what was measured, on
which data, by which metric, or against what baseline. Off by default:

```bash
SO_MLFLOW_ENABLED=true uv run uvicorn app.main:app
uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db
```

A run records the parameters (including the **seed**), the **git commit**, the
configured data source, the **data vintage** from the store, forecast metrics as
mean/min/max, every strategy's performance under its own prefix (`HRP__sharpe`), and
per-ticker tables as artifacts.

### The point, in two logged rows

Two pipeline runs over AAPL/MSFT/JPM/PFE, 2018–2022:

| Backend | `mase_vs_naive_mean` | `legacy_approximate_accuracy_mean` |
|---|---:|---:|
| `lightgbm` | 1.0085 | 98.16 |
| `naive` | **1.0000** | 98.19 |

The legacy metric puts LightGBM at 98.16 against a do-nothing baseline of 98.19 —
indistinguishable. Both sit on the same run, so neither can be quoted without the
other. `directional_accuracy_mean` was 0.5012, a coin flip.

**On the size of that gap.** Re-running the same configuration across five seeds gives
mean MASE between 1.0042 and 1.0232 — a spread of 0.0190, wider than the 0.0129 by
which the model trails the baseline. The correct statement is that LightGBM is
**indistinguishable from doing nothing**, not that it is worse. Quoting that gap as a
finding would be the same mistake as quoting 93%, in the opposite direction.

### Tracking never breaks the work

A run is telemetry. If the store is unreachable or a metric will not serialise, the
pipeline warns and finishes with `tracking_run_id: null`. Three tests assert this,
including one pointing the tracking URI at an uncreatable path. A `None` metric is
skipped rather than logged as zero — `directional_accuracy` is null for a flat
forecast, and zero would read as "always wrong" instead of "never guessed".

---

## Packaging

Two files replace the old ad-hoc setup:

| Was | Is now |
|---|---|
| `requirements.txt` — version *floors*, resolved fresh on every install | `pyproject.toml` + `uv.lock` — 211 packages pinned with hashes |
| `pytest.ini` | `[tool.pytest.ini_options]` in `pyproject.toml` |

Four dependency sets, so a deployment installs what it will actually run:

| Set | Install | Holds | Why it is separate |
|---|---|---|---|
| runtime | `uv sync --no-dev` | 173 packages: API, jobs, Polars/DuckDB, optimizers, PyTorch, LightGBM, MLflow | what the image ships |
| `keras` extra | `uv sync --extra keras` | `tensorflow-cpu` | 1.3 GB installed, for the one backend the [measured table](#every-backend-measured-on-real-data) shows is *worse* than a random walk. Absent, it disappears from `/health` and nothing else changes |
| `dev` group | `uv sync` (the default) | `pytest`, `httpx`, `fakeredis`, `redislite`, `ruff` | never shipped |

`uv sync --all-extras` is the full set — the one the measured table was produced
on, since it needs `keras_lstm`. The front end has its own dependencies and its
own lockfile under `frontend/`; no Python install pulls them.

### The lockfile is the point

`uv.lock` is committed, `frontend/package-lock.json` beside it, and every
install path that matters passes `--locked` (or `npm ci`),
which **fails** on a stale lockfile rather than quietly resolving something else.
So the image cannot contain a dependency set that was never tested, and a
reviewer can see the exact set in the diff.

### torch comes from the CPU index

Every model here trains on CPU, but `pip install torch` does not know that. The
default PyPI build resolves to 29 packages, 16 of them `nvidia-*` and `triton`
wheels. This lockfile contains **none** of them, because `[tool.uv.sources]`
routes torch to PyTorch's CPU index on Linux and Windows:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
```

macOS is excluded on purpose: there is no separate CPU wheel there, the default
PyPI build already being CPU-only. Previously this was a comment in
`requirements.txt` telling you to pass `--index-url` by hand, which nobody does.

### Two commands, one image

```toml
[project.scripts]
frontier = "app.cli:run"              # the pipeline CLI, was `python main.py`
frontier-worker = "app.worker:run"    # the RQ worker, was `python -m app.worker`
```

Both old invocations still work: `main.py` is a shim over `app/cli.py`, and
`python -m app.worker` is unchanged.

`Dockerfile` builds one image that runs either role, because an API and a worker
that were built separately would eventually execute different code. It is
multi-stage:

- dependencies install in a layer keyed on `uv.lock` alone, so editing source
  does not re-download torch;
- the project installs `--no-editable`, so the runtime stage copies the virtual
  environment and no source tree;
- runtime is `python:3.12-slim` plus `libgomp1` (LightGBM links against it; the
  torch wheel ships its own), running as uid 10001, not root;
- `HEALTHCHECK` probes `/health` with `urllib`, so the image needs no `curl`.

The image is **3.9 GB**, which is large for a service and mostly not this
project's code: torch is 769 MB even as a CPU build, and `riskfolio-lib` pulls
`vectorbt` → `numba`/`llvmlite` (240 MB), `astropy` (60 MB) and `statsmodels`,
none of which the optimizers here call directly. Bytecode is precompiled
(`UV_COMPILE_BYTECODE=1`), trading some of that size for faster cold starts.

`compose.yaml` wires the UI, the API, a worker and Redis:

```bash
docker compose up --build                  # UI on http://127.0.0.1:8080
docker compose up -d --scale worker=4      # forecasting is CPU-bound; scale by process
docker compose run --rm api frontier --tickers AAPL MSFT --backend lightgbm
docker build --build-arg EXTRAS="--extra keras" -t frontier:keras .
```

Three choices there are deliberate. `SO_JOB_BACKEND=redis`, not `auto`: in a
container the broker is always supposed to be there, so falling back to the
in-process queue would hide a broken deployment behind a working `/health`. The
API and worker share one named volume, because the worker *writes* the Parquet
store and the MLflow database that the API then reads. And only the `api`
service declares the build for the Python image — two services building one tag
race each other and collide when tagging, which is exactly what the first
`docker compose build` did.

---

## Front end

React 19 + TypeScript on Vite, charts in ECharts, six pages against the same API
the CLI uses. `npm run dev` proxies to the service; `docker compose up` serves
the built bundle from nginx on <http://127.0.0.1:8080>.

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
`tsc`, not a browser. Re-generate after changing anything in `app/schemas/`:

```bash
uv run python scripts/gen_api_types.py
```

It already caught two things. `PipelineResult.warnings` and `.failed` have
defaults server-side but are *not* required in the schema, so the pages that
rendered them unconditionally would have thrown on a response that omitted them.
And the result endpoint published no schema at all for what a finished job
contains — `JobStatus.result` was typed `Any`, since the job store is generic
and keeps whatever a task returns. `app/schemas/job_results.py` now names the
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

---

## Quality

Nothing below is advice: CI runs all of it on every push, and the badge-less
truth is that a red build blocks nothing but your own confidence — this is a
single-maintainer repo, so the gates exist to catch what review misses.

```bash
uv run ruff check .        # lint
uv run ruff format .       # format
uv run pytest --cov        # 321 tests, coverage floor 84%
cd frontend
npm run lint               # biome: lint + format, one tool
npm run typecheck          # tsc
npm test                   # 21 jsdom tests
npm run test:e2e           # 4 Playwright tests against a running stack
```

### One linter per language, with reasons in the config

`ruff` for Python, `biome` for TypeScript. Both lint *and* format, so style is
never a review topic. The rule sets are explicit rather than inherited: a
default set that changes under you is how a clean build turns red on an upgrade
you did not make.

Four rules are switched off, each with its reason in `pyproject.toml` — imports
inside functions (`app/tasks.py` defers heavy imports so a worker pays for them
only when it runs that task), magic-value comparisons (`assert mase > 0.9` is
what a numerical test looks like), argument counts (a backtest genuinely takes
eight parameters), and one that fights the formatter. Everything else is on.

The interesting category is the 12 blind `except Exception` handlers, which are
load-bearing: one ticker that will not train must not sink a request, one
strategy that fails must not drop the other nine, and tracking must never fail
the work it is recording. Each now carries a `# noqa: BLE001` naming that
reason, so the rule keeps working on new code while the deliberate ones are
documented where they sit.

### The front end is linted by Biome, not ESLint

`typescript-eslint`'s peer range stops at TypeScript 6 and this project is on 7,
so ESLint cannot parse what actually compiles. Biome does not depend on the
TypeScript package at all. Its hooks rule immediately paid for itself: Phase 9
left two `eslint-disable` comments for a linter that was never installed, and
both were hiding real dependency-array bugs — `slotOf` was recreated every
render and silently excluded from two memos. It is a `useCallback` now.

The generated `src/api/schema.ts` is excluded from both lint and format: it is
regenerated from the API's OpenAPI schema, and any edit would be undone.

### Coverage, and what it does not say

85.9% under branch coverage, floor at 84. `app/worker.py` reads 0% and is left
in the measurement anyway — `tests/test_worker_process.py` runs it as a real
subprocess, which coverage does not follow, and omitting it would raise the
number without covering a line.

### What CI actually checks

| Job | Catches |
|---|---|
| `python` | lint, format, 321 tests, the coverage floor |
| `frontend` | biome, `tsc`, 21 unit tests, a production build |
| `types-are-current` | a Pydantic model changed without regenerating the front end's types — which would otherwise fail in a browser, not a build |
| `images` | both Dockerfiles build, the stack comes up healthy, `/health` reports the Redis backend |
| `e2e` | Playwright drives a real browser through nginx → API → Redis → worker: it submits a pipeline run, waits for the worker to finish it, and asserts a canvas was painted and the numbers rendered |

`SO_MARKET_DATA_SOURCE=synthetic` throughout: CI must never depend on yfinance,
which rate-limits and restates history.

Writing these found two real defects. The e2e suite caught the first on its
first run: Portfolio and Backtest disabled their run button below two tickers
without saying why, while Run explained itself — a dead control with no
explanation. Both pages say it now.

The second is worse and would have shipped. `proxy_pass http://api:8000`
resolves the upstream **once, at startup**, so recreating the API container left
the UI serving 502 until nginx was restarted too — a restart, a redeploy, or a
`compose up` after an image change, all of which are routine. nginx now
re-resolves through Docker's DNS per request, and the `images` job restarts the
API and re-checks the UI so the bug cannot come back.

### Hooks

`uv run pre-commit install` once per clone gets ruff and biome with `--fix` on
staged files, plus the usual merge-conflict and large-file checks — the last of
which exists because a 361 MB virtualenv was once committed to this repo.
Everything the hooks do, CI does again; skipping a hook costs a CI run, not
correctness.

---

## Known limitations

The migration is finished, so nothing here is waiting on a later phase. These
are the things this project does not do, written down so the next person does
not have to discover them:

- **`POST /portfolio/optimize` and `POST /pipeline/runs` are in-sample.** They fit
  weights on the window they report on, which is useful for inspecting a single
  allocation but is not an out-of-sample result. Use
  [`POST /backtest/runs`](#walk-forward-backtest) for that.
- **The backtest uses realised returns, not forecasts.** It measures the
  *strategies*, holding the forecast fixed. Walking the forecasting models forward
  too — retraining at each rebalance — would be considerably more expensive and is
  not implemented; `BacktestRequest.on_forecast` is reserved for it.
- **Costs are a flat spread.** `cost_bps` on traded notional, with no market impact,
  no bid-ask modelling and no borrow cost on shorts.
- **The store keeps only the latest value per bar.** Revisions are *reported*, and
  `ingested_at` records which batch wrote each surviving row, but the superseded
  value is not retained — so you cannot replay a backtest against last month's
  vintage. Full bitemporal history would mean keeping every version of every bar.
- **`/data/query` uses a keyword denylist, not a SQL parser.** It is deliberately
  conservative and will refuse some harmless queries. Do not expose it to untrusted
  callers on the strength of that check alone.
- **The store holds volume and intraday range, but no feature uses them yet.** The
  Phase 3 features are close-only; widening them is a follow-on.
- **No model artifacts are logged.** MLflow can store a trained model and its
  signature; runs here record parameters, metrics and tables only. Re-running with
  the logged seed and parameters reproduces the model rather than loading it.
- **`mlflow` is a heavy dependency** — it pulls SQLAlchemy, Flask and Alembic for the
  database backend and the UI. `mlflow-skinny` is lighter but cannot use a database
  backend, and MLflow 3 has put the filesystem backend into maintenance mode, so the
  full package is the only combination that works without running a server.
- **Only jobs are tracked.** `POST /forecast` and `POST /portfolio/optimize` are
  inspection endpoints and log nothing; pipeline runs and backtests are the
  experiments.
- **One-step-ahead only.** Every backend predicts the next trading day. Multi-horizon
  forecasting is not implemented.
- **Sequence models are memory-bound at scale.** A 60-step window over 22 features
  is 60× the source data. Returning a strided view defers that cost, but PyTorch
  materialises each batch, so a very large universe still needs batched loading
  rather than one array.
- **`Gerber_InvVar` still reads only the diagonal.** It now uses Riskfolio's
  published Gerber statistic rather than the old sign-based approximation, but
  inverse-variance weighting ignores the off-diagonals the statistic exists to
  estimate. Kept under its original name for continuity; use `Gerber_HRP` instead.
- **Risk tolerance is a positional pick** along the volatility ranking, not a
  utility-maximising choice.
- **HERC is not offered.** Riskfolio 7.3.0 raises `TypeError` from its own
  `_hierarchical_recursive_bisection` for that model regardless of arguments.
- **Five superseded optimizer modules remain** — `mean_variance.py`,
  `risk_parity.py`, `gmv.py`, `hrp.py` and `gerber.py` are no longer called by
  anything. They are kept because their tests encode the defects found during the
  audit; delete them when that history stops being useful.
- **The end-to-end suite is four tests, on one browser.** Chromium only, and it
  covers the pipeline path — submit, wait for the worker, read the numbers —
  plus a deep link and a guard. Forecast, Data and Tracking are not driven in a
  browser by anything, and no test looks at a rendered chart's *content*: a
  canvas with the wrong line on it passes.
- **The bundle is 900 kB** (295 kB gzipped), 596 kB of it ECharts. It is split
  into its own chunk so it caches independently, but nothing is lazy-loaded by
  route: opening Tracking still pays for the chart library.
- **`/data/query` results are rendered as text.** Every column is right-aligned
  and stringified, because the API returns untyped SQL rows. Fine for the
  aggregate queries it is meant for; not a spreadsheet.
- **The lint rule set is a judgement call.** Four rule families are off, with
  reasons in `pyproject.toml`, and 12 blind excepts carry per-line suppressions.
  A stricter reading of any of them is defensible; what is not defensible is
  turning a rule off silently, which is why each has a comment.
- **Coverage measures lines, not behaviour.** 85.9% with a floor at 84 says the
  tests execute most of the code. It says nothing about whether the assertions
  are the right ones, and the number would barely move if half of them were
  deleted.
- **The risk-tolerance slider re-selects client-side.** It re-picks along the
  volatility ranking with the same positional rule the service uses, so moving
  it never re-runs anything — but it is a second implementation of that rule,
  and the two could drift.
- **`venv/` is a 361 MB Windows virtualenv left over from the original project.**
  It is no longer tracked by git, and `.gitignore` keeps it that way, but it is
  still sitting in the working directory: `rm -rf venv/` when you want the disk
  back. It contains only numpy, pandas and streamlit — not TensorFlow, scipy,
  scikit-learn or yfinance — so it could never have run this project.
- **CI has never run.** The workflow is written and every step was executed
  locally — `uv sync --locked`, ruff, 321 tests against the coverage floor,
  biome, `tsc`, 21 unit tests, both image builds, the stack coming up healthy,
  and the four Playwright tests against it — but GitHub has not run it once, so
  the YAML itself is unproven. The first push is the test.
- **The API image is 3.9 GB.** See [Packaging](#packaging) for where it goes.
  Most of it is inherited from `riskfolio-lib`'s dependency tree and the torch
  wheel, so trimming it means dropping capability, not tidying. The UI image is
  81 MB.
- **The Dockerfile avoids BuildKit-only syntax** (cache and bind mounts), so it
  builds with the classic builder where `buildx` is not installed. Rebuilds
  re-download wheels that a cache mount would have kept.
- **The image pins no base digest.** It tracks `python:3.12-slim-bookworm` by
  tag, so two builds a month apart can differ below the Python layer even with
  an unchanged `uv.lock`. Pin the digest if you need bit-identical rebuilds.
- **`requires-python` is capped at `<3.13`.** The `keras` extra has no 3.13+
  wheels, and the rest is untested there; the cap is honesty about what was run,
  not a known incompatibility.
- **Cancelling a running job needs a real worker.** `DELETE /pipeline/runs/{id}`
  stops a running job only on the `redis` backend; on `memory` it can cancel a
  queued job but only flags a running one, because Python threads cannot be
  interrupted.

### Data sources

`SO_MARKET_DATA_SOURCE` selects one of three:

- **`yfinance`** (default) — hits the network on every request. Convenient, and not
  reproducible, because upstream restates history.
- **`parquet`** — reads the local store. The reproducible option; see
  [Price store](#price-store).
- **`synthetic`** — deterministic geometric random walks seeded per ticker
  (`Layer1_Preprocessing/synthetic.py`). Useful for
development without a network, for reproducible demos, and for tests whose work
runs in a separate process. It is also a sanity check: a model that appears to
beat a random walk on this data has a bug, because there is no structure to find.
Phase 6 adds a `parquet` source alongside it.

---

## Migration roadmap

| Phase | Area | Change | Status |
|-------|------|--------|--------|
| 1 | API | FastAPI + Pydantic | **done** |
| 2 | Jobs | RQ + Redis | **done** |
| 3 | Forecasting | PyTorch + LightGBM | **done** |
| 4 | Optimizers | cvxpy (Clarabel/OSQP) + Riskfolio-Lib | **done** |
| 5 | Dataframes | Polars | **done** |
| 6 | Storage | Parquet + DuckDB | **done** |
| 7 | Tracking | MLflow | **done** |
| 8 | Packaging | uv + pyproject.toml + Docker | **done** |
| 9 | Frontend | React + TypeScript + Vite, ECharts | **done** |
| 10 | Quality | pytest + ruff + git | **done** |
