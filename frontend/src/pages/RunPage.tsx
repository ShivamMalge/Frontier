/**
 * The whole pipeline: prices in, weights and performance out.
 *
 * The run is a job, not a request -- `keras_lstm` trains one model per ticker
 * and takes minutes each -- so this submits, polls, and renders when it lands.
 */
import { useCallback, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import { series, weightsFor } from "../api/frame";
import type { PipelineResult, StrategyPerformance } from "../api/schema";
import { Chart } from "../charts/Chart";
import { bars, timeSeries } from "../charts/options";
import { barChartClass, MAX_SERIES, makeSlots, usePalette } from "../charts/theme";
import { Card, Chips, Field, Note, Stat } from "../components/Bits";
import { JobBar } from "../components/JobBar";
import { RunControls } from "../components/RunControls";
import { type Column, Signed, Table } from "../components/Table";
import { useJob } from "../hooks/useJob";
import { num, pct } from "../lib/format";
import { useMeta } from "../state";

/** The same positional rule the service uses, applied without re-running. */
function chooseByRisk(performance: StrategyPerformance[], tolerance: number): string {
  if (!performance.length) return "";
  const ranked = [...performance].sort((a, b) => a.annual_volatility - b.annual_volatility);
  const position = Math.round(Math.min(Math.max(tolerance, 0), 1) * (ranked.length - 1));
  return ranked[position]?.strategy ?? "";
}

/** Green only when the *displayed* value beats the baseline: a raw 0.9996
    rounds to "1.000", and colouring that green next to a true 1.000 reads as a
    rendering bug. */
function beatsBaseline(mase: number): boolean {
  return Number(num(mase)) < 1;
}

function isPipelineResult(value: unknown): value is PipelineResult {
  return !!value && typeof value === "object" && "forecast_metrics" in value && "weights" in value;
}

export function RunPage() {
  const { params } = useMeta();
  const palette = usePalette();
  const [risk, setRisk] = useState(0.5);
  const [compare, setCompare] = useState<string[]>([]);

  const job = useJob<PipelineResult>((result) => (isPipelineResult(result) ? result : null));
  const result = job.result;

  const performance = result?.performance ?? [];
  const selected = result ? chooseByRisk(performance, risk) : "";

  // There are ten strategies and eight hues, so a fixed catalogue index would
  // hand two strategies the same colour. Slots are allocated to what is on
  // screen instead, and an allocated slot stays with its strategy until that
  // strategy leaves the chart -- so toggling one never repaints the others.
  const allocate = useRef(makeSlots());

  const shown = useMemo(() => {
    if (!result) return [];
    if (compare.length) return compare.slice(0, MAX_SERIES);
    // Default view: the risk-selected strategy plus the best by Sharpe, capped
    // at the palette. More than that is spaghetti; the table carries the rest.
    const bySharpe = [...performance].sort((a, b) => b.sharpe - a.sharpe).map((p) => p.strategy);
    return [selected, ...bySharpe.filter((name) => name !== selected)].slice(0, 4);
  }, [result, compare, performance, selected]);

  const slots = useMemo(() => {
    const active = Array.from(new Set([...shown, selected].filter(Boolean)));
    const assigned: Record<string, number> = {};
    for (const name of active) assigned[name] = allocate.current(name, active);
    return assigned;
  }, [shown, selected]);
  const slotOf = useCallback((name: string) => slots[name] ?? 0, [slots]);

  const growth = useMemo(() => {
    if (!result) return null;
    return timeSeries(
      palette,
      shown.map((strategy) => ({
        name: strategy,
        points: series(result.cumulative_growth, strategy),
        slot: slotOf(strategy),
      })),
      { format: (v) => num(v, 2), zoom: true, label: "cumulative growth" },
    );
  }, [result, shown, palette, slotOf]);

  const mase = useMemo(() => {
    if (!result) return null;
    const rows: [string, number][] = result.forecast_metrics.map((m) => [
      m.ticker,
      m.mase_vs_naive,
    ]);
    return bars(palette, rows, {
      slot: 0,
      format: (v) => num(v, 3),
      reference: { value: 1, label: "random walk" },
      label: "MASE by ticker",
    });
  }, [result, palette]);

  const weightsChart = useMemo(() => {
    if (!result || !selected) return null;
    return bars(palette, weightsFor(result.weights, selected), {
      slot: slotOf(selected),
      format: (v) => pct(v, 1),
      label: `weights for ${selected}`,
    });
  }, [result, selected, palette, slotOf]);

  const metricColumns: Column<PipelineResult["forecast_metrics"][number]>[] = [
    {
      key: "ticker",
      header: "Ticker",
      render: (m) => <span className="mono">{m.ticker}</span>,
    },
    {
      key: "mase",
      header: "MASE",
      num: true,
      render: (m) => (
        <span className={beatsBaseline(m.mase_vs_naive) ? "pos" : undefined}>
          {num(m.mase_vs_naive)}
        </span>
      ),
    },
    {
      key: "dir",
      header: "Directional",
      num: true,
      render: (m) => (m.directional_accuracy == null ? "n/a" : pct(m.directional_accuracy, 1)),
    },
    { key: "rmse", header: "RMSE", num: true, render: (m) => num(m.rmse, 2) },
    { key: "r2", header: "R²", num: true, render: (m) => num(m.r2) },
    {
      key: "legacy",
      header: "Legacy “accuracy”",
      num: true,
      render: (m) => (
        <span style={{ color: "var(--text-muted)" }}>{num(m.legacy_approximate_accuracy, 2)}</span>
      ),
    },
  ];

  const performanceColumns: Column<StrategyPerformance>[] = [
    { key: "strategy", header: "Strategy", render: (p) => p.strategy },
    {
      key: "ret",
      header: "Return",
      num: true,
      render: (p) => <Signed value={p.annual_return} render={(v) => pct(v)} />,
    },
    {
      key: "vol",
      header: "Volatility",
      num: true,
      render: (p) => pct(p.annual_volatility),
    },
    {
      key: "sharpe",
      header: "Sharpe",
      num: true,
      render: (p) => <Signed value={p.sharpe} render={(v) => num(v)} />,
    },
    {
      key: "sortino",
      header: "Sortino",
      num: true,
      render: (p) => num(p.sortino),
    },
    {
      key: "dd",
      header: "Max drawdown",
      num: true,
      render: (p) => <span className="neg">{pct(p.max_drawdown)}</span>,
    },
  ];

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Run the pipeline</h1>
          <p>
            Forecast each ticker, turn the forecasts into expected returns, optimize every strategy
            on them and compare the results. Submitted as a job you can watch.
          </p>
        </div>
      </div>

      <Card>
        <RunControls>
          <Field
            label={`Risk tolerance ${risk.toFixed(2)}`}
            hint="0 = lowest volatility, 1 = highest"
          >
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={risk}
              onChange={(event) => setRisk(Number(event.target.value))}
            />
          </Field>
          <button
            type="button"
            className="primary"
            disabled={job.busy || params.tickers.length < 2}
            onClick={() =>
              job.submit(() =>
                api.submitPipeline({
                  tickers: params.tickers,
                  start: params.start,
                  end: params.end,
                  backend: params.backend,
                  risk_tolerance: risk,
                }),
              )
            }
          >
            {job.busy ? "Running…" : "Run pipeline"}
          </button>
        </RunControls>

        {params.tickers.length < 2 && <Note kind="warn">Select at least two tickers.</Note>}
        <div style={{ marginTop: 14 }}>
          <JobBar {...job} onCancel={job.busy ? job.cancel : undefined} />
        </div>
        {job.error && <Note kind="error">{job.error.message}</Note>}
      </Card>

      {result && (
        <>
          {(result.warnings ?? []).map((warning) => (
            <Note kind="warn" key={warning}>
              {warning}
            </Note>
          ))}
          {(result.failed ?? []).length > 0 && (
            <Note kind="warn">
              Skipped: {(result.failed ?? []).map((f) => `${f.ticker} (${f.reason})`).join(", ")}
            </Note>
          )}

          <Card>
            <dl className="grid stats" style={{ margin: 0 }}>
              <Stat
                label="Backend"
                value={<span style={{ fontSize: 15 }}>{result.backend}</span>}
              />
              <Stat label="Tickers" value={result.tickers.length} />
              <Stat
                label={`Selected at ${risk.toFixed(2)}`}
                value={<span style={{ fontSize: 15 }}>{selected}</span>}
              />
              <Stat
                label="Its Sharpe"
                value={num(performance.find((p) => p.strategy === selected)?.sharpe ?? null)}
              />
            </dl>
            <p
              style={{
                color: "var(--text-muted)",
                fontSize: 12,
                marginTop: 10,
                marginBottom: 0,
              }}
            >
              The slider re-picks along the volatility ranking without re-running anything — the
              same positional rule the service applies.
            </p>
          </Card>

          <div className="grid two">
            <Card
              title="Forecast quality"
              hint="MASE is the model's error divided by a random walk's. Below 1.0 beats doing nothing."
            >
              {mase && (
                <Chart
                  option={mase}
                  className={barChartClass(result.forecast_metrics.length)}
                  label="MASE by ticker"
                />
              )}
              <Table
                rows={result.forecast_metrics}
                columns={metricColumns}
                rowKey={(m) => m.ticker}
              />
            </Card>

            <Card
              title={`Weights · ${selected}`}
              hint="Rounded to the nearest basis point in the table."
            >
              {weightsChart && (
                <Chart
                  option={weightsChart}
                  className={barChartClass(result.tickers.length)}
                  label={`weights for ${selected}`}
                />
              )}
              <Table
                rows={weightsFor(result.weights, selected).map(([ticker, weight]) => ({
                  ticker,
                  weight,
                }))}
                columns={[
                  {
                    key: "t",
                    header: "Ticker",
                    render: (r) => <span className="mono">{r.ticker}</span>,
                  },
                  {
                    key: "w",
                    header: "Weight",
                    num: true,
                    render: (r) => pct(r.weight),
                  },
                ]}
                rowKey={(r) => r.ticker}
              />
            </Card>
          </div>

          <Card
            title="Cumulative growth of 1.0 invested"
            hint={`In-sample: weights are fitted on the window they are scored on. Showing ${shown.length} of ${performance.length} strategies.`}
            actions={
              <Chips
                options={performance.map((p) => p.strategy)}
                selected={shown}
                onToggle={(name) =>
                  setCompare((current) => {
                    const base = current.length ? current : shown;
                    return base.includes(name)
                      ? base.filter((other) => other !== name)
                      : [...base, name];
                  })
                }
                colorOf={(name) => `var(--series-${slotOf(name) + 1})`}
                disabledWhen={() =>
                  compare.length ? compare.length >= MAX_SERIES : shown.length >= MAX_SERIES
                }
              />
            }
          >
            {growth && (
              <Chart option={growth} className="chart tall" label="cumulative growth by strategy" />
            )}
          </Card>

          <Card
            title="Strategy performance"
            hint="Sorted by Sharpe. The highlighted row is the current selection."
          >
            <Table
              rows={[...performance].sort((a, b) => b.sharpe - a.sharpe)}
              columns={performanceColumns}
              rowKey={(p) => p.strategy}
              highlight={(p) => p.strategy === selected}
            />
          </Card>
        </>
      )}
    </>
  );
}
