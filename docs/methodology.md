# Methodology

What the numbers mean, and how much of the work each one is doing.
Start here before quoting a result from this project.

[← Back to the README](../README.md)

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

**Features** (`src/frontier/data/feature_engineering.py`, 22 columns): 10 lagged
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


---

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
