# Frontier

**LSTM forecasting & portfolio optimization.**

Forecast equity prices, optimize portfolios across six strategies, and compare
their risk-adjusted performance.

**Migration status: Phases 1–3 of 10 complete** — FastAPI + Pydantic service
layer, RQ + Redis job queue, PyTorch + LightGBM forecasting.
See [Migration roadmap](#migration-roadmap).

---

## Quick start

Requires Python 3.12 (TensorFlow has no 3.13+ wheels) and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Run the API:

```bash
.venv/bin/python -m uvicorn app.main:app --reload
```

Interactive docs at <http://127.0.0.1:8000/docs>.

Run a worker (needs Redis — see [Jobs](#jobs)):

```bash
docker run -d -p 6379:6379 redis:7-alpine     # or: sudo pacman -S redis && sudo systemctl start redis
.venv/bin/python -m app.worker
```

Without Redis the API still works: it falls back to an in-process queue and says
so in `GET /health`.

Run the CLI:

```bash
.venv/bin/python main.py --tickers AAPL MSFT JPM --backend naive
```

Run the tests:

```bash
.venv/bin/python -m pytest
```

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
| `POST` | `/api/v1/market/prices` | Adjusted closing prices |
| `POST` | `/api/v1/market/returns` | Simple daily returns |
| `GET` | `/api/v1/meta/backends` | Forecasting backends, scopes and availability |
| `POST` | `/api/v1/forecast` | Forecast prices, derive predicted returns |
| `POST` | `/api/v1/portfolio/optimize` | Weights for each strategy |
| `POST` | `/api/v1/portfolio/performance` | Risk/return statistics per strategy |
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
  `pytest` and `python main.py` working on a bare checkout. The fallback is logged
  as a warning, never silent.
- **`redis`** — require Redis; refuse to start without it. **Use this in
  production**, where quietly degrading to a single-process queue is worse than
  not booting.
- **`memory`** — never use Redis.

`GET /health` reports the backend actually in use.

```bash
SO_JOB_BACKEND=redis SO_REDIS_URL=redis://localhost:6379/0 \
  .venv/bin/python -m uvicorn app.main:app

.venv/bin/python -m app.worker --queues pipeline    # one per spare core
.venv/bin/python -m app.worker --burst              # drain and exit (CI)
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
  jobs/               job execution (Phase 2)
    base.py           JobStore protocol, JobRecord, TaskRef
    memory.py         thread-pool store
    redis_store.py    RQ + Redis store
    progress.py       backend-agnostic progress reporting
  tasks.py            functions workers execute, by dotted path
  worker.py           RQ worker entry point
  settings.py         env-overridable config
  errors.py           domain exceptions -> HTTP

Layer1_Preprocessing/ price retrieval, return computation, synthetic source
Layer1_LSTM/          forecasting (name is historical, not LSTM-only)
    torch_lstm.py     PyTorch LSTM
    gbm.py            LightGBM
    datasets.py       leak-free supervised splits
    scaling.py        scalers fitted on training data only
    results.py        SeriesForecast
    train_lstm.py     legacy Keras path
Layer2_Optimization/  the six optimizers + dispatch
Layer3_Portfolio_Generation/  construction, performance, selection
Layer4_Visualization/ dead matplotlib code -- superseded, see below
Layer5_Streamlit_App/ interim UI (Phase 9 replaces it)
utils/                config, logging, filesystem helpers
tests/                123 tests, no network access
```

Dependencies point one way: `app` → `Layer*` → `utils`. Nothing in the numerical
core imports from `app`, so the CLI, the API and the Streamlit app all share one
implementation and cannot drift apart.

---

## Strategies

| Strategy | Family | Long-only | Uses return forecast |
|----------|--------|-----------|----------------------|
| `Markowitz_MaxSharpe` | mean-variance | yes | **yes** |
| `Markowitz_MinVar` | mean-variance | yes | no |
| `RiskParity` | risk-based | yes | no |
| `GMV` | mean-variance | no | no |
| `HRP` | risk-based | yes | no |
| `Gerber_InvVar` | robust-covariance | yes | no |

Worth knowing: only max-Sharpe uses the forecast at all. The other five depend
solely on the covariance matrix, so a better forecasting model cannot improve
them. `GMV` and `Markowitz_MinVar` are the same objective — the only difference
is that GMV permits short positions.

---

## Known limitations

Carried forward deliberately, each scheduled to a later phase:

- **In-sample optimization** — weights are derived from the forecast over the
  same window used to score them. There is no walk-forward backtest. *(Phase 4)*
- **No transaction costs or turnover limits.** Weights are treated as free to
  reach. *(Phase 4)*
- **One-step-ahead only.** Every backend predicts the next trading day. Multi-horizon
  forecasting is not implemented.
- **`Gerber_InvVar` discards most of its work** — `gerber_covariance` builds a
  full matrix in a `d²` Python loop, then inverse-variance weighting reads only
  the diagonal. The implementation also is not the published Gerber statistic,
  which thresholds at ±c·σ rather than on raw signs. *(Phase 4, via Riskfolio-Lib)*
- **Risk tolerance is a positional pick** along the volatility ranking, not a
  utility-maximising choice.
- **`Layer4_Visualization/` is dead code** — four matplotlib `plt.show()`
  functions, imported by nothing, and `plot_effiecient_frontier.py` (sic) plots
  cumulative growth rather than an efficient frontier. Phase 9 supersedes it with
  ECharts. It is left in place rather than deleted; remove it when you are ready.
- **`venv/` is a 361 MB Windows virtualenv committed into the repo**, containing
  only numpy, pandas and streamlit — not TensorFlow, scipy, scikit-learn or
  yfinance, so it could never have run this project. Delete it:
  `rm -rf venv/`. *(Phase 8 formalises packaging)*
- **No git history.** This directory is not a repository. `git init` is Phase 10.
- **Cancelling a running job needs a real worker.** `DELETE /pipeline/runs/{id}`
  stops a running job only on the `redis` backend; on `memory` it can cancel a
  queued job but only flags a running one, because Python threads cannot be
  interrupted.

### Offline data source

`SO_MARKET_DATA_SOURCE=synthetic` replaces yfinance with deterministic geometric
random walks seeded per ticker (`Layer1_Preprocessing/synthetic.py`). Useful for
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
| 4 | Optimizers | cvxpy (Clarabel/OSQP) + Riskfolio-Lib | next |
| 5 | Dataframes | Polars | |
| 6 | Storage | Parquet + DuckDB | |
| 7 | Tracking | MLflow | |
| 8 | Packaging | uv + pyproject.toml + Docker | |
| 9 | Frontend | React + TypeScript + Vite, ECharts | |
| 10 | Quality | pytest + ruff + git | |
