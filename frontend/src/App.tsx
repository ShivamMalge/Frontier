import { Navigate, NavLink, Route, Routes } from "react-router-dom";
import { Note } from "./components/Bits";
import { BacktestPage } from "./pages/BacktestPage";
import { DataPage } from "./pages/DataPage";
import { ForecastPage } from "./pages/ForecastPage";
import { PortfolioPage } from "./pages/PortfolioPage";
import { RunPage } from "./pages/RunPage";
import { TrackingPage } from "./pages/TrackingPage";
import { useMeta } from "./state";

const PAGES = [
  { to: "/run", label: "Run", hint: "pipeline" },
  { to: "/forecast", label: "Forecast", hint: "models" },
  { to: "/portfolio", label: "Portfolio", hint: "weights" },
  { to: "/backtest", label: "Backtest", hint: "walk-forward" },
  { to: "/data", label: "Data", hint: "store" },
  { to: "/tracking", label: "Tracking", hint: "mlflow" },
];

function Health() {
  const { health, offline, loading } = useMeta();
  if (loading) return <div className="status-line">connecting…</div>;
  if (offline || !health) {
    return (
      <div className="status-line">
        <span>
          <span className="dot bad" />
          API unreachable
        </span>
      </div>
    );
  }
  return (
    <div
      className="status-line"
      style={{ flexDirection: "column", alignItems: "flex-start", gap: 4 }}
    >
      <span>
        <span className="dot ok" />
        {health.app} {health.version}
      </span>
      <span className="mono" style={{ fontSize: 11 }}>
        jobs: {health.job_backend}
      </span>
      <span className="mono" style={{ fontSize: 11 }}>
        backends: {health.forecast_backends.length}
      </span>
    </div>
  );
}

export function App() {
  const { offline, loading } = useMeta();
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <b>Frontier</b>
          <span>v0.1</span>
        </div>
        <nav className="nav">
          {PAGES.map((page) => (
            <NavLink
              key={page.to}
              to={page.to}
              className={({ isActive }) => (isActive ? "active" : "")}
            >
              {page.label}
              <small>{page.hint}</small>
            </NavLink>
          ))}
        </nav>
        <div style={{ marginTop: "auto" }}>
          <Health />
          <p style={{ color: "var(--text-muted)", fontSize: 11, marginTop: 10 }}>
            MASE below 1.0 beats a random walk. Directional accuracy of 0.5 is a coin flip.
          </p>
        </div>
      </aside>

      <main className="main">
        {!loading && offline && (
          <Note kind="error">
            The API is not answering. Start it with{" "}
            <code>uv run uvicorn app.main:app --reload</code> and reload this page.
          </Note>
        )}
        <Routes>
          <Route path="/" element={<Navigate to="/run" replace />} />
          <Route path="/run" element={<RunPage />} />
          <Route path="/forecast" element={<ForecastPage />} />
          <Route path="/portfolio" element={<PortfolioPage />} />
          <Route path="/backtest" element={<BacktestPage />} />
          <Route path="/data" element={<DataPage />} />
          <Route path="/tracking" element={<TrackingPage />} />
          <Route path="*" element={<Note kind="warn">No such page.</Note>} />
        </Routes>
      </main>
    </div>
  );
}
