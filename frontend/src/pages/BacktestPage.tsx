/**
 * The walk-forward backtest: the only out-of-sample number in the app.
 *
 * Everything on the Run and Portfolio pages is fitted on the window it reports
 * on. Here the optimizer sees only the trailing `lookback` days at each
 * rebalance, and trading is charged for.
 */
import { useMemo, useState } from "react";
import { api } from "../api/client";
import { lastValue, series } from "../api/frame";
import type { BacktestResponse } from "../api/schema";
import { Chart } from "../charts/Chart";
import { bars, timeSeries } from "../charts/options";
import { barChartClass, usePalette } from "../charts/theme";
import { Card, Field, Note, Stat } from "../components/Bits";
import { JobBar } from "../components/JobBar";
import { RunControls } from "../components/RunControls";
import { type Column, Signed, Table } from "../components/Table";
import { useJob } from "../hooks/useJob";
import { num, pct } from "../lib/format";
import { useMeta } from "../state";

function isBacktest(value: unknown): value is BacktestResponse {
  return !!value && typeof value === "object" && "rebalance_dates" in value && "results" in value;
}

export function BacktestPage() {
  const { params } = useMeta();
  const palette = usePalette();
  const [lookback, setLookback] = useState(252);
  const [rebalance, setRebalance] = useState(21);
  const [costBps, setCostBps] = useState(10);
  const [strategy, setStrategy] = useState<string | null>(null);

  const job = useJob<BacktestResponse>((result) => (isBacktest(result) ? result : null));
  const result = job.result;

  const best = useMemo(() => {
    if (!result) return null;
    return [...result.results].sort((a, b) => b.sharpe - a.sharpe)[0] ?? null;
  }, [result]);

  const chosen =
    strategy && result?.cumulative_growth.columns.includes(strategy)
      ? strategy
      : (best?.strategy ?? null);

  // Net and gross are the same quantity for the same strategy, so they belong
  // on one axis: the gap between them is the cost drag, read directly.
  const growth = useMemo(() => {
    if (!result || !chosen) return null;
    return timeSeries(
      palette,
      [
        {
          name: `${chosen} · net`,
          points: series(result.cumulative_growth, chosen),
          slot: 0,
        },
        {
          name: `${chosen} · gross`,
          points: series(result.gross_cumulative_growth, chosen),
          slot: 1,
          dashed: true,
        },
      ],
      {
        format: (v) => num(v, 3),
        zoom: true,
        label: "net against gross growth",
      },
    );
  }, [result, chosen, palette]);

  const dragChart = useMemo(() => {
    if (!result) return null;
    const rows = [...result.results]
      .sort((a, b) => b.cost_drag - a.cost_drag)
      .map((r): [string, number] => [r.strategy, r.cost_drag]);
    return bars(palette, rows, {
      slot: 1,
      format: (v) => pct(v, 2),
      label: "cost drag by strategy",
    });
  }, [result, palette]);

  const columns: Column<BacktestResponse["results"][number]>[] = [
    { key: "s", header: "Strategy", render: (r) => r.strategy },
    {
      key: "net",
      header: "Net return",
      num: true,
      render: (r) => <Signed value={r.annual_return} render={(v) => pct(v)} />,
    },
    {
      key: "gross",
      header: "Gross",
      num: true,
      render: (r) => pct(r.gross_annual_return),
    },
    {
      key: "drag",
      header: "Cost drag",
      num: true,
      render: (r) => <span className="neg">{pct(r.cost_drag)}</span>,
    },
    {
      key: "vol",
      header: "Volatility",
      num: true,
      render: (r) => pct(r.annual_volatility),
    },
    {
      key: "sh",
      header: "Sharpe",
      num: true,
      render: (r) => <Signed value={r.sharpe} render={(v) => num(v)} />,
    },
    {
      key: "dd",
      header: "Max drawdown",
      num: true,
      render: (r) => <span className="neg">{pct(r.max_drawdown)}</span>,
    },
    {
      key: "to",
      header: "Turnover / yr",
      num: true,
      render: (r) => pct(r.annual_turnover, 0),
    },
    { key: "rb", header: "Rebalances", num: true, render: (r) => r.rebalances },
  ];

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Walk-forward backtest</h1>
          <p>
            Re-optimises on a trailing window and holds forward through unseen data, charging{" "}
            <code>cost_bps</code> on traded notional. Measures the strategies, holding the forecast
            fixed.
          </p>
        </div>
      </div>

      <Card>
        <RunControls showBackend={false}>
          <Field label="Lookback" hint="trading days per fit">
            <input
              type="number"
              min={60}
              step={21}
              value={lookback}
              onChange={(e) => setLookback(Number(e.target.value))}
            />
          </Field>
          <Field label="Rebalance every" hint="trading days">
            <input
              type="number"
              min={1}
              step={1}
              value={rebalance}
              onChange={(e) => setRebalance(Number(e.target.value))}
            />
          </Field>
          <Field label="Cost (bps)" hint="on traded notional">
            <input
              type="number"
              min={0}
              step={1}
              value={costBps}
              onChange={(e) => setCostBps(Number(e.target.value))}
            />
          </Field>
          <button
            type="button"
            className="primary"
            disabled={job.busy || params.tickers.length < 2}
            onClick={() =>
              job.submit(() =>
                api.submitBacktest({
                  tickers: params.tickers,
                  start: params.start,
                  end: params.end,
                  lookback,
                  rebalance_every: rebalance,
                  cost_bps: costBps,
                }),
              )
            }
          >
            {job.busy ? "Running…" : "Run backtest"}
          </button>
        </RunControls>
        {params.tickers.length < 2 && (
          <Note kind="warn">Select at least two tickers: a covariance needs a pair.</Note>
        )}
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

          <Card>
            <dl className="grid stats" style={{ margin: 0 }}>
              <Stat label="Out-of-sample days" value={result.observations} />
              <Stat
                label="Rebalances"
                value={result.rebalance_dates.length}
                sub={`every ${result.rebalance_every}d`}
              />
              <Stat
                label="Best by Sharpe"
                value={<span style={{ fontSize: 15 }}>{best?.strategy ?? "--"}</span>}
                sub={num(best?.sharpe ?? null)}
              />
              <Stat
                label="Its growth"
                value={num(lastValue(result.cumulative_growth, best?.strategy ?? ""), 3)}
                sub="net of costs"
              />
            </dl>
          </Card>

          <Card
            title="Net against gross"
            hint="The gap is what trading cost. Costs are a flat spread: no market impact, no borrow."
            actions={
              <select value={chosen ?? ""} onChange={(event) => setStrategy(event.target.value)}>
                {result.cumulative_growth.columns.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            }
          >
            {growth && (
              <Chart
                option={growth}
                className="chart tall"
                label="net against gross cumulative growth"
              />
            )}
          </Card>

          <Card title="Cost drag" hint="Gross annual return minus net, by strategy.">
            {dragChart && (
              <Chart
                option={dragChart}
                className={barChartClass(result.results.length)}
                label="cost drag by strategy"
              />
            )}
          </Card>

          <Card title="Out-of-sample results" hint="Sorted by Sharpe. Click a row to chart it.">
            <Table
              rows={[...result.results].sort((a, b) => b.sharpe - a.sharpe)}
              columns={columns}
              rowKey={(r) => r.strategy}
              highlight={(r) => r.strategy === chosen}
              onSelect={(r) => setStrategy(r.strategy)}
            />
          </Card>
        </>
      )}
    </>
  );
}
