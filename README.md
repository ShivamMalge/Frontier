# Frontier

**LSTM forecasting & portfolio optimization.**

Forecast equity prices, optimize portfolios across six strategies, and compare
their risk-adjusted performance.

**Migration status: Phases 1–8 of 10 complete**, plus walk-forward backtesting —
FastAPI + Pydantic service layer, RQ + Redis job queue, PyTorch + LightGBM
forecasting, cvxpy + Riskfolio-Lib optimizers, Polars data pipeline, Parquet +
DuckDB price store, MLflow tracking, uv + Docker packaging. See
[Migration roadmap](#migration-roadmap).

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
uv run pytest
```

Or bring up API, worker and Redis together:

```bash
docker compose up --build
```

`uv run <cmd>` syncs the environment first, so it is always in step with
`uv.lock`. `.venv/bin/<cmd>` works too once `uv sync` has run. Two dependency
sets are optional and left out by default — the legacy TensorFlow backend
(`--extra keras`) and the interim Streamlit UI (`--extra ui`); see
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
Layer4_Visualization/ dead matplotlib code -- superseded, see below
Layer5_Streamlit_App/ interim UI (Phase 9 replaces it)
utils/                config, logging, filesystem helpers
tests/                321 tests, no network access

pyproject.toml        dependencies, extras, entry points, pytest config
uv.lock               the resolved set, committed -- 212 packages
Dockerfile            one image, two roles (API and worker)
compose.yaml          API + worker + Redis
main.py               shim so `python main.py` works without an install
```

Dependencies point one way: `app` → `Layer*` → `utils`. Nothing in the numerical
core imports from `app`, so the CLI, the API and the Streamlit app all share one
implementation and cannot drift apart.

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
| `requirements.txt` — version *floors*, resolved fresh on every install | `pyproject.toml` + `uv.lock` — 212 packages pinned with hashes |
| `pytest.ini` | `[tool.pytest.ini_options]` in `pyproject.toml` |

Four dependency sets, so a deployment installs what it will actually run:

| Set | Install | Holds | Why it is separate |
|---|---|---|---|
| runtime | `uv sync --no-dev` | 173 packages: API, jobs, Polars/DuckDB, optimizers, PyTorch, LightGBM, MLflow | what the image ships |
| `keras` extra | `uv sync --extra keras` | `tensorflow-cpu` | 1.3 GB installed, for the one backend the [measured table](#every-backend-measured-on-real-data) shows is *worse* than a random walk. Absent, it disappears from `/health` and nothing else changes |
| `ui` extra | `uv sync --extra ui` | `streamlit` | interim UI; Phase 9 deletes it |
| `dev` group | `uv sync` (the default) | `pytest`, `httpx`, `fakeredis`, `redislite`, `ruff` | never shipped |

`uv sync --all-extras` is the full set — the one the measured table was produced
on, since it needs `keras_lstm`.

### The lockfile is the point

`uv.lock` is committed, and every install path that matters passes `--locked`,
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

`compose.yaml` wires API, worker and Redis:

```bash
docker compose up --build                  # API on http://127.0.0.1:8000
docker compose up -d --scale worker=4      # forecasting is CPU-bound; scale by process
docker compose run --rm api frontier --tickers AAPL MSFT --backend lightgbm
docker build --build-arg EXTRAS="--extra keras" -t frontier:keras .
```

Two choices there are deliberate. `SO_JOB_BACKEND=redis`, not `auto`: in a
container the broker is always supposed to be there, so falling back to the
in-process queue would hide a broken deployment behind a working `/health`. And
the API and worker share one named volume, because the worker *writes* the
Parquet store and the MLflow database that the API then reads.

---

## Known limitations

Carried forward deliberately, each scheduled to a later phase:

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
- **`Layer4_Visualization/` is dead code** — four matplotlib `plt.show()`
  functions, imported by nothing, and `plot_effiecient_frontier.py` (sic) plots
  cumulative growth rather than an efficient frontier. Phase 9 supersedes it with
  ECharts. It is left in place rather than deleted; remove it when you are ready.
- **`venv/` is a 361 MB Windows virtualenv left over from the original project.**
  It is no longer tracked by git, and `.gitignore` keeps it that way, but it is
  still sitting in the working directory: `rm -rf venv/` when you want the disk
  back. It contains only numpy, pandas and streamlit — not TensorFlow, scipy,
  scikit-learn or yfinance — so it could never have run this project.
- **No CI builds the image.** It was built and run end to end by hand — API
  healthy, worker executing a queued pipeline run, `keras_lstm` correctly absent
  from `/health` — but nothing re-checks that on a change, so a break would
  surface on somebody's next `docker compose up`. CI is Phase 10.
- **The image is 3.9 GB.** See [Packaging](#packaging) for where it goes. Most of
  it is inherited from `riskfolio-lib`'s dependency tree and the torch wheel, so
  trimming it means dropping capability, not tidying.
- **The Dockerfile avoids BuildKit-only syntax** (cache and bind mounts), so it
  builds with the classic builder where `buildx` is not installed. Rebuilds
  re-download wheels that a cache mount would have kept.
- **The image pins no base digest.** It tracks `python:3.12-slim-bookworm` by
  tag, so two builds a month apart can differ below the Python layer even with
  an unchanged `uv.lock`. Pin the digest if you need bit-identical rebuilds.
- **`requires-python` is capped at `<3.13`.** The `keras` extra has no 3.13+
  wheels, and the rest is untested there; the cap is honesty about what was run,
  not a known incompatibility.
- **The Streamlit app is not in the wheel or the image.** `Layer5_Streamlit_App/`
  and `Layer4_Visualization/` have no `__init__.py` and are imported by nothing,
  so they stay in the checkout. Phase 9 replaces both.
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
| 9 | Frontend | React + TypeScript + Vite, ECharts | next |
| 10 | Quality | pytest + ruff + git | |
