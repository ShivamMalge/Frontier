# Roadmap and limitations

What this does not do, and the ten phases that got it here.

[← Back to the README](../README.md)

---

## Known limitations

The migration is finished, so nothing here is waiting on a later phase. These
are the things this project does not do, written down so the next person does
not have to discover them:

- **`POST /portfolio/optimize` and `POST /pipeline/runs` are in-sample.** They fit
  weights on the window they report on, which is useful for inspecting a single
  allocation but is not an out-of-sample result. Use
  [`POST /backtest/runs`](methodology.md#walk-forward-backtest) for that.
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
- **The API image is 3.9 GB.** See [Packaging](operations.md#packaging) for where it goes.
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
  [Price store](architecture.md#price-store).
- **`synthetic`** — deterministic geometric random walks seeded per ticker
  (`src/frontier/data/synthetic.py`). Useful for
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
