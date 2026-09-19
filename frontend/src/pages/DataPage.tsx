/**
 * The Parquet price store: what it holds, where it has gaps, and SQL over it.
 *
 * The store is the reproducible data source -- yfinance restates history, so
 * only a stored snapshot gives repeatable results.
 */
import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { CoverageResponse, IngestResponse, QueryResponse } from "../api/schema";
import { useAsync } from "../hooks/useAsync";
import { useJob } from "../hooks/useJob";
import { useMeta } from "../state";
import { Card, Note, Stat } from "../components/Bits";
import { RunControls } from "../components/RunControls";
import { JobBar } from "../components/JobBar";
import { Table, type Column } from "../components/Table";
import { Chart } from "../charts/Chart";
import { bars } from "../charts/options";
import { barChartClass, usePalette } from "../charts/theme";
import { count } from "../lib/format";

const SAMPLE_SQL = "SELECT ticker, count(*) AS rows, min(date) AS first, max(date) AS last\nFROM prices GROUP BY ticker ORDER BY rows DESC";

function isIngest(value: unknown): value is IngestResponse {
  return !!value && typeof value === "object" && "rows_written" in value;
}

export function DataPage() {
  const { params } = useMeta();
  const palette = usePalette();
  const coverage = useAsync<CoverageResponse>();
  const query = useAsync<QueryResponse>();
  const [sql, setSql] = useState(SAMPLE_SQL);
  const ingest = useJob<IngestResponse>((result) => (isIngest(result) ? result : null));

  const { run: runCoverage } = coverage;
  useEffect(() => {
    void runCoverage((signal) => api.coverage(signal));
  }, [runCoverage]);

  // Re-read coverage when an ingest finishes: the store just changed.
  useEffect(() => {
    if (ingest.state === "succeeded") void runCoverage((signal) => api.coverage(signal));
  }, [ingest.state, runCoverage]);

  const store = coverage.data;

  const rowsChart = useMemo(() => {
    if (!store?.coverage.length) return null;
    const rows = [...store.coverage]
      .sort((a, b) => b.rows - a.rows)
      .map((c): [string, number] => [c.ticker, c.rows]);
    return bars(palette, rows, { slot: 2, format: (v) => count(v), label: "rows per ticker" });
  }, [store, palette]);

  const coverageColumns: Column<CoverageResponse["coverage"][number]>[] = [
    { key: "t", header: "Ticker", render: (c) => <span className="mono">{c.ticker}</span> },
    { key: "rows", header: "Rows", num: true, render: (c) => count(c.rows) },
    { key: "first", header: "First", num: true, render: (c) => c.first_date },
    { key: "last", header: "Last", num: true, render: (c) => c.last_date },
    { key: "vint", header: "Vintages", num: true, render: (c) => c.vintages },
    { key: "ing", header: "Last ingested", num: true, render: (c) => c.last_ingested.slice(0, 19).replace("T", " ") },
  ];

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Price store</h1>
          <p>
            Parquet on disk, queried with DuckDB. Ingesting records a vintage, so a revised bar is
            reported rather than silently overwritten.
          </p>
        </div>
      </div>

      <Card title="Ingest" hint="Downloads OHLCV for the selected universe into the store. Runs as a job.">
        <RunControls showBackend={false}>
          <button
            type="button"
            className="primary"
            disabled={ingest.busy || params.tickers.length < 1}
            onClick={() =>
              ingest.submit(() =>
                api.submitIngest({ tickers: params.tickers, start: params.start, end: params.end }),
              )
            }
          >
            {ingest.busy ? "Ingesting…" : "Ingest"}
          </button>
        </RunControls>
        <div style={{ marginTop: 14 }}>
          <JobBar {...ingest} onCancel={ingest.busy ? ingest.cancel : undefined} />
        </div>
        {ingest.error && <Note kind="error">{ingest.error.message}</Note>}
        {ingest.result && (
          <dl className="grid stats" style={{ marginTop: 14, marginBottom: 0 }}>
            <Stat label="Rows written" value={count(ingest.result.rows_written)} />
            <Stat label="Added" value={count(ingest.result.rows_added)} />
            <Stat label="Revised" value={count(ingest.result.rows_revised)} sub={ingest.result.revised_tickers.join(" ")} />
          </dl>
        )}
      </Card>

      {coverage.error && <Note kind="error">{coverage.error.message}</Note>}

      {store && !store.exists && (
        <Note kind="info">
          No store yet at <code>{store.store_root}</code>. Ingest above to create it.
        </Note>
      )}

      {store?.exists && (
        <>
          <Card title="Coverage" hint={`Stored at ${store.store_root}`}>
            <dl className="grid stats" style={{ marginBottom: 14 }}>
              <Stat label="Tickers" value={store.tickers.length} />
              <Stat label="Rows" value={count(store.total_rows)} />
              <Stat label="Gaps" value={store.gaps.length} sub="runs of missing days" />
            </dl>
            {rowsChart && <Chart option={rowsChart} className={barChartClass(store.coverage.length)} label="rows per ticker" />}
            <Table rows={store.coverage} columns={coverageColumns} rowKey={(c) => c.ticker} />
          </Card>

          {store.gaps.length > 0 && (
            <Card title="Gaps" hint="Runs of missing trading days, longest first.">
              <Table
                rows={[...store.gaps].sort((a, b) => b.gap_days - a.gap_days).slice(0, 25)}
                columns={[
                  { key: "t", header: "Ticker", render: (g) => <span className="mono">{g.ticker}</span> },
                  { key: "s", header: "From", num: true, render: (g) => g.gap_start },
                  { key: "e", header: "To", num: true, render: (g) => g.gap_end },
                  { key: "d", header: "Days", num: true, render: (g) => g.gap_days },
                ]}
                rowKey={(g) => `${g.ticker}-${g.gap_start}`}
              />
            </Card>
          )}
        </>
      )}

      <Card
        title="Query"
        hint="Read-only SQL against the store. The service checks against a keyword denylist, not a SQL parser — it will refuse some harmless queries."
      >
        <textarea value={sql} onChange={(event) => setSql(event.target.value)} spellCheck={false} />
        <div style={{ display: "flex", gap: 10, marginTop: 10 }}>
          <button
            type="button"
            className="primary"
            disabled={query.loading}
            onClick={() => void query.run((signal) => api.query({ sql, limit: 200 }, signal))}
          >
            {query.loading ? "Running…" : "Run query"}
          </button>
          <button type="button" className="ghost" onClick={() => setSql(SAMPLE_SQL)}>
            Reset
          </button>
        </div>
        {query.error && <Note kind="error">{query.error.message}</Note>}
        {query.data && (
          <div style={{ marginTop: 14 }}>
            {query.data.truncated && <Note kind="warn">Truncated to {query.data.row_count} rows.</Note>}
            <Table
              rows={query.data.rows.map((row, index) => ({ row, index }))}
              columns={query.data.columns.map((column, position) => ({
                key: `${column}-${position}`,
                header: column,
                num: true,
                render: (entry: { row: unknown[] }) => String(entry.row[position] ?? ""),
              }))}
              rowKey={(entry) => String(entry.index)}
              empty="No rows."
            />
          </div>
        )}
      </Card>
    </>
  );
}
