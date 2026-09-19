/**
 * MLflow runs, read through the API rather than the MLflow UI.
 *
 * Tracking is off by default and never fails a run: when it is off this page
 * says so and tells you how to turn it on, rather than showing an error.
 */
import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { TrackedRun, TrackedRunList, TrackingStatus } from "../api/schema";
import { Card, Note, Stat } from "../components/Bits";
import { type Column, Table } from "../components/Table";
import { useAsync } from "../hooks/useAsync";
import { num, timestamp } from "../lib/format";

const METRIC_KEYS = ["mase_mean", "sharpe_best", "mase_min", "mase_max"];

export function TrackingPage() {
  const status = useAsync<TrackingStatus>();
  const runs = useAsync<TrackedRunList>();
  const [limit, setLimit] = useState(25);
  const [open, setOpen] = useState<string | null>(null);

  const { run: runStatus } = status;
  const { run: runRuns } = runs;
  useEffect(() => {
    void runStatus((signal) => api.trackingStatus(signal));
  }, [runStatus]);
  useEffect(() => {
    void runRuns((signal) => api.trackingRuns(limit, signal));
  }, [runRuns, limit]);

  const enabled = status.data?.enabled ?? false;
  const rows = runs.data?.runs ?? [];
  const selected = useMemo(() => rows.find((r) => r.run_id === open) ?? null, [rows, open]);

  const columns: Column<TrackedRun>[] = [
    {
      key: "name",
      header: "Run",
      render: (r) => <span className="mono">{r.run_name ?? r.run_id.slice(0, 8)}</span>,
    },
    { key: "kind", header: "Kind", render: (r) => r.kind ?? "--" },
    { key: "status", header: "Status", render: (r) => r.status },
    {
      key: "started",
      header: "Started",
      num: true,
      render: (r) => timestamp(r.started_at),
    },
    {
      key: "commit",
      header: "Commit",
      num: true,
      render: (r) => <span className="mono">{r.git_commit?.slice(0, 8) ?? "--"}</span>,
    },
    {
      key: "vintage",
      header: "Data vintage",
      num: true,
      render: (r) => r.data_vintage?.slice(0, 10) ?? "--",
    },
    ...METRIC_KEYS.map(
      (key): Column<TrackedRun> => ({
        key,
        header: key.replace(/_/g, " "),
        num: true,
        render: (r) => num((r.metrics?.[key] as number | undefined) ?? null),
      }),
    ),
  ];

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Tracking</h1>
          <p>
            Every pipeline run and backtest records its parameters, its seed, the git commit and the
            data vintage — so a number can be traced back to what produced it.
          </p>
        </div>
      </div>

      {status.data && !enabled && (
        <Note kind="info">
          Tracking is off. Start the API with <code>SO_MLFLOW_ENABLED=true</code> to record runs.
          {status.data.detail ? ` ${status.data.detail}` : ""}
        </Note>
      )}

      {status.data && enabled && (
        <Card title="Status">
          <dl className="grid stats" style={{ margin: 0 }}>
            <Stat
              label="Experiment"
              value={<span style={{ fontSize: 15 }}>{status.data.experiment}</span>}
            />
            <Stat label="Runs" value={status.data.run_count ?? 0} />
            <Stat
              label="Store"
              value={
                <span style={{ fontSize: 12 }} className="mono">
                  {status.data.tracking_uri}
                </span>
              }
            />
          </dl>
          {status.data.ui_command && (
            <p
              style={{
                marginTop: 12,
                marginBottom: 0,
                color: "var(--text-secondary)",
              }}
            >
              Browse them in MLflow's own UI: <code>{status.data.ui_command}</code>
            </p>
          )}
        </Card>
      )}

      <Card
        title="Recent runs"
        hint="Click a run to see every parameter and metric it recorded."
        actions={
          <select value={limit} onChange={(event) => setLimit(Number(event.target.value))}>
            {[10, 25, 50, 100].map((value) => (
              <option key={value} value={value}>
                last {value}
              </option>
            ))}
          </select>
        }
      >
        {runs.error && <Note kind="error">{runs.error.message}</Note>}
        <Table
          rows={rows}
          columns={columns}
          rowKey={(r) => r.run_id}
          highlight={(r) => r.run_id === open}
          onSelect={(r) => setOpen(r.run_id === open ? null : r.run_id)}
          empty={
            enabled ? "No runs recorded yet." : "Tracking is off, so there is nothing to show."
          }
        />
      </Card>

      {selected && (
        <div className="grid two">
          <Card title="Parameters">
            <Table
              rows={Object.entries(selected.params ?? {}).map(([key, value]) => ({
                key,
                value: String(value),
              }))}
              columns={[
                {
                  key: "k",
                  header: "Name",
                  render: (p) => <span className="mono">{p.key}</span>,
                },
                {
                  key: "v",
                  header: "Value",
                  num: true,
                  render: (p) => p.value,
                },
              ]}
              rowKey={(p) => p.key}
              empty="No parameters recorded."
            />
          </Card>
          <Card title="Metrics">
            <Table
              rows={Object.entries(selected.metrics ?? {}).map(([key, value]) => ({
                key,
                value: Number(value),
              }))}
              columns={[
                {
                  key: "k",
                  header: "Name",
                  render: (m) => <span className="mono">{m.key}</span>,
                },
                {
                  key: "v",
                  header: "Value",
                  num: true,
                  render: (m) => num(m.value, 4),
                },
              ]}
              rowKey={(m) => m.key}
              empty="No metrics recorded."
            />
          </Card>
        </div>
      )}
    </>
  );
}
