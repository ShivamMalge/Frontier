/**
 * One forecast, inspected closely.
 *
 * `POST /forecast` is synchronous, so this page is for the fast backends and
 * for looking at a single ticker's fit -- not for training eighteen LSTMs.
 */
import { useMemo, useState } from "react";
import { api } from "../api/client";
import type { ForecastResponse, ForecastTarget } from "../api/schema";
import { series } from "../api/frame";
import { useAsync } from "../hooks/useAsync";
import { useMeta } from "../state";
import { Card, Field, Note, Spinner, Stat } from "../components/Bits";
import { RunControls } from "../components/RunControls";
import { Table, type Column } from "../components/Table";
import { Chart } from "../charts/Chart";
import { bars, timeSeries } from "../charts/options";
import { barChartClass, usePalette } from "../charts/theme";
import { num, pct } from "../lib/format";

/** Green only when the *displayed* value beats the baseline: a raw 0.9996
    rounds to "1.000", and colouring that green next to a true 1.000 reads as a
    rendering bug. */
function beatsBaseline(mase: number): boolean {
  return Number(num(mase)) < 1;
}

export function ForecastPage() {
  const { params, backends } = useMeta();
  const palette = usePalette();
  const [target, setTarget] = useState<ForecastTarget>("return");
  const [multiSeries, setMultiSeries] = useState(false);
  const [focus, setFocus] = useState<string | null>(null);
  const request = useAsync<ForecastResponse>();
  const result = request.data;

  const scope = backends.find((b) => b.name === params.backend)?.scope;
  const ticker = focus && result?.tickers.includes(focus) ? focus : (result?.tickers[0] ?? null);

  const fit = useMemo(() => {
    if (!result || !ticker) return null;
    // Actual and predicted are the same quantity, so they share one axis.
    return timeSeries(
      palette,
      [
        { name: "actual", points: series(result.actual_prices, ticker), slot: 0 },
        { name: "predicted", points: series(result.predicted_prices, ticker), slot: 1, dashed: true },
      ],
      { format: (v) => num(v, 2), zoom: true, label: `${ticker} actual vs predicted` },
    );
  }, [result, ticker, palette]);

  const maseChart = useMemo(() => {
    if (!result) return null;
    return bars(
      palette,
      result.metrics.map((m): [string, number] => [m.ticker, m.mase_vs_naive]),
      {
        format: (v) => num(v, 3),
        reference: { value: 1, label: "random walk" },
        label: "MASE by ticker",
      },
    );
  }, [result, palette]);

  const columns: Column<ForecastResponse["metrics"][number]>[] = [
    { key: "ticker", header: "Ticker", render: (m) => <span className="mono">{m.ticker}</span> },
    { key: "obs", header: "Observations", num: true, render: (m) => m.observations },
    {
      key: "mase",
      header: "MASE",
      num: true,
      render: (m) => <span className={beatsBaseline(m.mase_vs_naive) ? "pos" : undefined}>{num(m.mase_vs_naive)}</span>,
    },
    {
      key: "dir",
      header: "Directional",
      num: true,
      render: (m) => (m.directional_accuracy == null ? "n/a" : pct(m.directional_accuracy, 1)),
    },
    { key: "rmse", header: "RMSE", num: true, render: (m) => num(m.rmse, 2) },
    { key: "mae", header: "MAE", num: true, render: (m) => num(m.mae, 2) },
    { key: "mape", header: "MAPE", num: true, render: (m) => pct(m.mape / 100, 2) },
    { key: "r2", header: "R²", num: true, render: (m) => num(m.r2) },
  ];

  const worst = result?.metrics.reduce(
    (a, b) => (b.mase_vs_naive > (a?.mase_vs_naive ?? -Infinity) ? b : a),
    result.metrics[0],
  );

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Forecast</h1>
          <p>
            Models emit price levels; optimizers consume returns. This page shows both, and the
            metrics that say whether the model beat doing nothing.
          </p>
        </div>
      </div>

      <Card>
        <RunControls>
          <Field label="Target" hint="A return target is what makes the models work at all.">
            <select value={target} onChange={(event) => setTarget(event.target.value as ForecastTarget)}>
              <option value="return">return</option>
              <option value="price">price</option>
            </select>
          </Field>
          <Field label="Scope" hint={scope === "universe" ? "Backend can pool tickers." : "Per-ticker backend."}>
            <select
              value={multiSeries ? "universe" : "ticker"}
              disabled={scope !== "universe"}
              onChange={(event) => setMultiSeries(event.target.value === "universe")}
            >
              <option value="ticker">one model per ticker</option>
              <option value="universe">one shared model</option>
            </select>
          </Field>
          <button
            type="button"
            className="primary"
            disabled={request.loading || params.tickers.length < 1}
            onClick={() =>
              void request.run((signal) =>
                api.forecast(
                  {
                    tickers: params.tickers,
                    start: params.start,
                    end: params.end,
                    backend: params.backend,
                    target,
                    multi_series: multiSeries,
                  },
                  signal,
                ),
              )
            }
          >
            {request.loading ? "Forecasting…" : "Forecast"}
          </button>
        </RunControls>
        {request.loading && (
          <div style={{ marginTop: 12 }}>
            <Spinner>
              This runs inside the request. `keras_lstm` takes minutes per ticker — use the Run page
              for that.
            </Spinner>
          </div>
        )}
        {request.error && <Note kind="error">{request.error.message}</Note>}
      </Card>

      {result && (
        <>
          {(result.failed ?? []).length > 0 && (
            <Note kind="warn">
              Skipped: {(result.failed ?? []).map((f) => `${f.ticker} (${f.reason})`).join(", ")}
            </Note>
          )}

          <Card>
            <dl className="grid stats" style={{ margin: 0 }}>
              <Stat label="Backend" value={<span style={{ fontSize: 15 }}>{result.backend}</span>} sub={result.target} />
              <Stat label="Lookback" value={result.lookback_window} sub="days" />
              <Stat label="Train split" value={pct(result.train_split, 0)} />
              <Stat label="Worst MASE" value={num(worst?.mase_vs_naive ?? null)} sub={worst?.ticker} />
            </dl>
          </Card>

          <Card
            title="Predicted against actual"
            hint="Price levels. A one-step-ahead forecast that tracks the line closely is usually just repeating yesterday's price."
            actions={
              <select value={ticker ?? ""} onChange={(event) => setFocus(event.target.value)}>
                {result.tickers.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            }
          >
            {fit && <Chart option={fit} className="chart tall" label={`${ticker} actual against predicted`} />}
          </Card>

          <Card title="Forecast quality" hint="The two columns that carry signal are MASE and directional accuracy.">
            {maseChart && <Chart option={maseChart} className={barChartClass(result.metrics.length)} label="MASE by ticker" />}
            <Table rows={result.metrics} columns={columns} rowKey={(m) => m.ticker} />
          </Card>
        </>
      )}
    </>
  );
}
