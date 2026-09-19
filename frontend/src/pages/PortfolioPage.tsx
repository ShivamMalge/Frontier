/**
 * Optimize on realised returns, then trace the frontier.
 *
 * This page deliberately feeds the optimizers *realised* returns from
 * `/market/returns` rather than forecasts: it is for comparing the strategies
 * themselves. The Run page is the one that puts a forecast in front of them.
 */
import { useCallback, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import { series, weightsFor } from "../api/frame";
import type {
  Frame,
  FrontierResponse,
  OptimizeResponse,
  PerformanceResponse,
  StrategyPerformance,
} from "../api/schema";
import { Chart } from "../charts/Chart";
import { bars, frontier as frontierOption, timeSeries } from "../charts/options";
import { makeSlots, usePalette } from "../charts/theme";
import { Card, Field, Note, Spinner, Stat } from "../components/Bits";
import { RunControls } from "../components/RunControls";
import { type Column, Signed, Table } from "../components/Table";
import { useAsync } from "../hooks/useAsync";
import { num, pct } from "../lib/format";
import { useMeta } from "../state";

interface Bundle {
  returns: Frame;
  optimize: OptimizeResponse;
  performance: PerformanceResponse;
  frontier: FrontierResponse;
}

export function PortfolioPage() {
  const { params } = useMeta();
  const palette = usePalette();
  const [minWeight, setMinWeight] = useState(0);
  const [maxWeight, setMaxWeight] = useState(1);
  const [strategy, setStrategy] = useState<string | null>(null);
  const request = useAsync<Bundle>();
  const data = request.data;

  // Ten strategies, eight hues: slots are allocated to what is charted rather
  // than indexed off the catalogue, which would collide. See RunPage.
  const allocate = useRef(makeSlots());

  const chosen =
    strategy && data?.optimize.weights.columns.includes(strategy)
      ? strategy
      : (data?.optimize.weights.columns[0] ?? null);

  async function run(signal: AbortSignal): Promise<Bundle> {
    const constraints = { min_weight: minWeight, max_weight: maxWeight };
    const returnsResponse = await api.returns(
      { tickers: params.tickers, start: params.start, end: params.end },
      signal,
    );
    const returns = returnsResponse.returns;
    const optimize = await api.optimize({ returns, constraints }, signal);
    // Performance needs both, so it cannot be merged into the call above.
    const [performance, frontierResult] = await Promise.all([
      api.performance({ returns, weights: optimize.weights }, signal),
      api.frontier({ returns, points: 24, constraints }, signal),
    ]);
    return { returns, optimize, performance, frontier: frontierResult };
  }

  const shown = useMemo(() => {
    if (!data) return [];
    return [...data.performance.performance]
      .sort((a, b) => b.sharpe - a.sharpe)
      .map((p) => p.strategy)
      .slice(0, 4);
  }, [data]);

  const slots = useMemo(() => {
    const active = Array.from(new Set([...shown, chosen].filter(Boolean))) as string[];
    const assigned: Record<string, number> = {};
    for (const name of active) assigned[name] = allocate.current(name, active);
    return assigned;
  }, [shown, chosen]);
  const slotOf = useCallback((name: string) => slots[name] ?? 0, [slots]);

  const growth = useMemo(() => {
    if (!data) return null;
    return timeSeries(
      palette,
      shown.map((name) => ({
        name,
        points: series(data.performance.cumulative_growth, name),
        slot: slotOf(name),
      })),
      { format: (v) => num(v, 2), zoom: true, label: "cumulative growth" },
    );
  }, [data, shown, palette, slotOf]);

  const frontierChart = useMemo(() => {
    if (!data) return null;
    // One mark, not several: the risk-based strategies land within a few basis
    // points of each other on a real frontier, and three labels on one point is
    // three labels nobody can read. Marking the selected strategy answers the
    // question the page is actually asking -- where does this one sit?
    const mark = data.performance.performance.find((p) => p.strategy === chosen);
    return frontierOption(
      palette,
      data.frontier.points,
      data.frontier.tangency_index,
      mark
        ? [
            {
              name: mark.strategy,
              volatility: mark.annual_volatility,
              expected_return: mark.annual_return,
            },
          ]
        : [],
    );
  }, [data, palette, chosen]);

  const weightsChart = useMemo(() => {
    if (!data || !chosen) return null;
    return bars(palette, weightsFor(data.optimize.weights, chosen), {
      slot: slotOf(chosen),
      format: (v) => pct(v, 1),
      label: `weights for ${chosen}`,
    });
  }, [data, chosen, palette, slotOf]);

  const performanceColumns: Column<StrategyPerformance>[] = [
    { key: "s", header: "Strategy", render: (p) => p.strategy },
    {
      key: "r",
      header: "Return",
      num: true,
      render: (p) => <Signed value={p.annual_return} render={(v) => pct(v)} />,
    },
    {
      key: "v",
      header: "Volatility",
      num: true,
      render: (p) => pct(p.annual_volatility),
    },
    {
      key: "sh",
      header: "Sharpe",
      num: true,
      render: (p) => <Signed value={p.sharpe} render={(v) => num(v)} />,
    },
    { key: "so", header: "Sortino", num: true, render: (p) => num(p.sortino) },
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
          <h1>Portfolio</h1>
          <p>
            Ten strategies on realised returns, with box constraints and the efficient frontier.
            In-sample: every weight here is fitted on the window it is scored on.
          </p>
        </div>
      </div>

      <Card>
        <RunControls showBackend={false}>
          <Field label="Min weight" hint="0 is long-only">
            <input
              type="number"
              step={0.05}
              min={-1}
              max={1}
              value={minWeight}
              onChange={(event) => setMinWeight(Number(event.target.value))}
            />
          </Field>
          <Field label="Max weight" hint="Cap per position">
            <input
              type="number"
              step={0.05}
              min={0}
              max={1}
              value={maxWeight}
              onChange={(event) => setMaxWeight(Number(event.target.value))}
            />
          </Field>
          <button
            type="button"
            className="primary"
            disabled={request.loading || params.tickers.length < 2}
            onClick={() => void request.run(run)}
          >
            {request.loading ? "Optimizing…" : "Optimize"}
          </button>
        </RunControls>
        {params.tickers.length < 2 && (
          <Note kind="warn">Select at least two tickers: a covariance needs a pair.</Note>
        )}
        {request.loading && (
          <div style={{ marginTop: 12 }}>
            <Spinner>Returns, ten optimizers, performance and a 24-point frontier.</Spinner>
          </div>
        )}
        {request.error && <Note kind="error">{request.error.message}</Note>}
      </Card>

      {data && (
        <>
          {(data.optimize.warnings ?? []).map((warning) => (
            <Note kind="warn" key={warning}>
              {warning}
            </Note>
          ))}

          <Card>
            <dl className="grid stats" style={{ margin: 0 }}>
              <Stat label="Observations" value={data.optimize.observations} sub="days" />
              <Stat label="Assets" value={data.optimize.tickers.length} />
              <Stat label="Risk-free" value={pct(data.optimize.risk_free_rate)} />
              <Stat
                label="Tangency Sharpe"
                value={num(data.frontier.points[data.frontier.tangency_index]?.sharpe ?? null)}
              />
            </dl>
          </Card>

          <div className="grid two">
            <Card
              title="Efficient frontier"
              hint={`Minimum-variance frontier from the same covariance the optimizers use. ${chosen ?? "The selected strategy"} is marked.`}
            >
              {frontierChart && (
                <Chart option={frontierChart} className="chart" label="efficient frontier" />
              )}
            </Card>

            <Card
              title={`Weights · ${chosen ?? ""}`}
              actions={
                <select value={chosen ?? ""} onChange={(event) => setStrategy(event.target.value)}>
                  {data.optimize.weights.columns.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              }
            >
              {weightsChart && (
                <Chart option={weightsChart} className="chart" label={`weights for ${chosen}`} />
              )}
            </Card>
          </div>

          <Card
            title="Cumulative growth of 1.0 invested"
            hint="The four best by Sharpe; the table below has all ten."
          >
            {growth && (
              <Chart option={growth} className="chart tall" label="cumulative growth by strategy" />
            )}
          </Card>

          <Card title="Performance" hint="Sorted by Sharpe.">
            <Table
              rows={[...data.performance.performance].sort((a, b) => b.sharpe - a.sharpe)}
              columns={performanceColumns}
              rowKey={(p) => p.strategy}
              highlight={(p) => p.strategy === chosen}
              onSelect={(p) => setStrategy(p.strategy)}
            />
          </Card>
        </>
      )}
    </>
  );
}
