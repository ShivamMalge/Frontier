/**
 * Mounting tests.
 *
 * These are not screenshots -- jsdom paints nothing. What they check is that
 * the app boots against a stubbed API, that the shell reports what /health
 * said, and that a finished pipeline job renders its numbers. That is the part
 * that breaks: a bad hook order or a null result reaching a table.
 */

import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { MetaProvider } from "./state";

// jsdom has no canvas, and a painted chart is not what these assert.
vi.mock("./charts/Chart", () => ({
  Chart: ({ label }: { label: string }) => <div data-testid="chart">{label}</div>,
}));

const UNIVERSE = {
  tickers: ["AAPL", "MSFT", "JPM"],
  default_start: "2015-01-02",
  default_end: "2022-12-30",
  lookback_window: 60,
  train_split: 0.67,
  risk_free_rate: 0.02,
  trading_days_per_year: 252,
};

const HEALTH = {
  status: "ok",
  app: "Frontier API",
  version: "0.1.0",
  forecast_backends: ["naive", "lightgbm"],
  job_backend: "redis",
};

const PIPELINE_RESULT = {
  tickers: ["AAPL", "MSFT"],
  failed: [],
  backend: "lightgbm",
  forecast_metrics: [
    {
      ticker: "AAPL",
      observations: 300,
      rmse: 2.5,
      mae: 1.9,
      r2: 0.98,
      mape: 1.4,
      directional_accuracy: null,
      mase_vs_naive: 0.987,
      legacy_approximate_accuracy: 98.2,
    },
  ],
  weights: {
    index: ["AAPL", "MSFT"],
    columns: ["HRP", "GMV"],
    data: [
      [0.6, 0.4],
      [0.4, 0.6],
    ],
  },
  performance: [
    {
      strategy: "HRP",
      annual_return: 0.12,
      annual_volatility: 0.18,
      sharpe: 0.67,
      sortino: 0.9,
      max_drawdown: -0.2,
    },
    {
      strategy: "GMV",
      annual_return: 0.08,
      annual_volatility: 0.14,
      sharpe: 0.57,
      sortino: 0.8,
      max_drawdown: -0.15,
    },
  ],
  cumulative_growth: {
    index: ["2022-01-03", "2022-01-04"],
    columns: ["HRP", "GMV"],
    data: [
      [1.0, 1.0],
      [1.01, 1.005],
    ],
  },
  selected_strategy: "GMV",
  selected_weights: { AAPL: 0.4, MSFT: 0.6 },
  warnings: [],
  diagnostics: {},
  tracking_run_id: null,
};

/** A fetch stub that answers by path, so no network is involved. */
function stubFetch(routes: Record<string, unknown>) {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const match = Object.keys(routes).find((path) => url.includes(path));
    if (!match) throw new Error(`unstubbed request: ${url}`);
    return {
      ok: true,
      status: 200,
      json: async () => routes[match],
    } as Response;
  });
}

function renderApp() {
  return render(
    <MemoryRouter initialEntries={["/run"]}>
      <MetaProvider>
        <App />
      </MetaProvider>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("the shell", () => {
  it("reports what /health said", async () => {
    vi.stubGlobal(
      "fetch",
      stubFetch({
        "/health": HEALTH,
        "/meta/universe": UNIVERSE,
        "/meta/strategies": { strategies: [] },
        "/meta/backends": { backends: [] },
      }),
    );
    renderApp();

    expect(await screen.findByText(/Frontier API 0.1.0/)).toBeTruthy();
    expect(screen.getByText(/jobs: redis/)).toBeTruthy();
  });

  it("says so when the API is unreachable instead of rendering empty controls", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("connection refused");
      }),
    );
    renderApp();

    expect(await screen.findByText(/API unreachable/)).toBeTruthy();
    expect(await screen.findByText(/The API is not answering/)).toBeTruthy();
  });
});

describe("a finished pipeline run", () => {
  it("renders its metrics, weights and performance", async () => {
    vi.stubGlobal(
      "fetch",
      stubFetch({
        "/health": HEALTH,
        "/meta/universe": UNIVERSE,
        "/meta/strategies": { strategies: [{ name: "HRP" }, { name: "GMV" }] },
        "/meta/backends": { backends: [] },
        "/pipeline/runs/job-1/result": {
          job_id: "job-1",
          state: "succeeded",
          created_at: "2026-01-01T00:00:00Z",
          progress: 1,
          result: PIPELINE_RESULT,
        },
        "/pipeline/runs/job-1": {
          job_id: "job-1",
          state: "succeeded",
          created_at: "2026-01-01T00:00:00Z",
          progress: 1,
        },
        "/pipeline/runs": {
          job_id: "job-1",
          state: "queued",
          status_url: "",
          result_url: "",
        },
      }),
    );
    renderApp();

    const button = await screen.findByRole("button", { name: /Run pipeline/ });
    button.click();

    // MASE, rendered to three decimals by the metrics table.
    expect(await screen.findByText("0.987")).toBeTruthy();
    // The null directional accuracy of a flat forecast must read "n/a", not NaN.
    expect(await screen.findByText("n/a")).toBeTruthy();
    await waitFor(() => expect(screen.getAllByTestId("chart").length).toBeGreaterThan(0));
  });
});
