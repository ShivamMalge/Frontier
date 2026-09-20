# Operations

Installing it, shipping it, tracking what it produced, and the checks
that run before any of that is believed.

[← Back to the README](../README.md)

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
| `keras` extra | `uv sync --extra keras` | `tensorflow-cpu` | 1.3 GB installed, for the one backend the [measured table](methodology.md#every-backend-measured-on-real-data) shows is *worse* than a random walk. Absent, it disappears from `/health` and nothing else changes |
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
frontier = "frontier.cli:run"              # the pipeline CLI
frontier-worker = "frontier.worker:run"    # the RQ worker
```

These are the entry points. The repository root used to carry a `main.py` shim
so `python main.py` worked in a checkout without an install; the src/ layout
retired it, because a module under `src/` is not importable from the working
directory by design. `python -m frontier.worker` still works.

`deploy/Dockerfile` builds one image that runs either role, because an API and a worker
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

`deploy/compose.yaml` wires the UI, the API, a worker and Redis. Both build
contexts are the repository root, so run it from there:

```bash
export COMPOSE_FILE=deploy/compose.yaml    # or pass -f deploy/compose.yaml each time

docker compose up --build                  # UI on http://127.0.0.1:8080
docker compose up -d --scale worker=4      # forecasting is CPU-bound; scale by process
docker compose run --rm api frontier --tickers AAPL MSFT --backend lightgbm

docker build -f deploy/Dockerfile --build-arg EXTRAS="--extra keras" -t frontier:keras .
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

## Experiment tracking

The original project reported "93%+ accuracy" with no record of what was measured, on
which data, by which metric, or against what baseline. Off by default:

```bash
SO_MLFLOW_ENABLED=true uv run uvicorn frontier.api.main:app
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
inside functions (`src/frontier/tasks.py` defers heavy imports so a worker pays for them
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

85.9% under branch coverage, floor at 84. `src/frontier/worker.py` reads 0% and is left
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
