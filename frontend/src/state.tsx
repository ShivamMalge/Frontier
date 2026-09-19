/**
 * Metadata and the run parameters shared across pages.
 *
 * The universe, date range and backend belong to the session, not to one page:
 * picking six tickers on Run and then opening Forecast should not mean picking
 * them again. They persist to localStorage so a reload keeps them too.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { api } from "./api/client";
import type {
  BackendInfo,
  ForecastBackend,
  HealthResponse,
  StrategyInfo,
  UniverseResponse,
} from "./api/schema";

export interface RunParams {
  tickers: string[];
  start: string;
  end: string;
  backend: ForecastBackend;
}

interface Meta {
  health: HealthResponse | null;
  universe: UniverseResponse | null;
  strategies: StrategyInfo[];
  backends: BackendInfo[];
  /** Null until the metadata calls have settled; an error means no API. */
  offline: boolean;
  loading: boolean;
  params: RunParams;
  setParams: (update: Partial<RunParams>) => void;
  toggleTicker: (ticker: string) => void;
}

const STORAGE_KEY = "frontier.params.v1";

const MetaContext = createContext<Meta | null>(null);

function readStored(): Partial<RunParams> {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Partial<RunParams>) : {};
  } catch {
    // A blocked or full localStorage is not a reason to fail the app.
    return {};
  }
}

export function MetaProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [universe, setUniverse] = useState<UniverseResponse | null>(null);
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [backends, setBackends] = useState<BackendInfo[]>([]);
  const [offline, setOffline] = useState(false);
  const [loading, setLoading] = useState(true);
  const [params, setParamsState] = useState<RunParams>(() => ({
    tickers: [],
    start: "",
    end: "",
    backend: "naive",
    ...readStored(),
  }));

  useEffect(() => {
    let live = true;
    (async () => {
      try {
        const [healthResponse, universeResponse, strategyResponse, backendResponse] =
          await Promise.all([api.health(), api.universe(), api.strategies(), api.backends()]);
        if (!live) return;
        setHealth(healthResponse);
        setUniverse(universeResponse);
        setStrategies(strategyResponse.strategies);
        setBackends(backendResponse.backends);
        setParamsState((current) => ({
          // Stored values win; the API's defaults fill the rest on first visit.
          tickers: current.tickers.length ? current.tickers : universeResponse.tickers.slice(0, 6),
          start: current.start || universeResponse.default_start,
          end: current.end || universeResponse.default_end,
          backend: current.backend,
        }));
        setOffline(false);
      } catch {
        if (live) setOffline(true);
      } finally {
        if (live) setLoading(false);
      }
    })();
    return () => {
      live = false;
    };
  }, []);

  const setParams = useCallback((update: Partial<RunParams>) => {
    setParamsState((current) => {
      const next = { ...current, ...update };
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch {
        // Per-browser convenience only; nothing here needs to persist.
      }
      return next;
    });
  }, []);

  const toggleTicker = useCallback(
    (ticker: string) => {
      setParamsState((current) => {
        const tickers = current.tickers.includes(ticker)
          ? current.tickers.filter((t) => t !== ticker)
          : [...current.tickers, ticker];
        const next = { ...current, tickers };
        try {
          window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
        } catch {
          /* ignore */
        }
        return next;
      });
    },
    [],
  );

  const value = useMemo(
    () => ({
      health,
      universe,
      strategies,
      backends,
      offline,
      loading,
      params,
      setParams,
      toggleTicker,
    }),
    [health, universe, strategies, backends, offline, loading, params, setParams, toggleTicker],
  );

  return <MetaContext.Provider value={value}>{children}</MetaContext.Provider>;
}

export function useMeta(): Meta {
  const value = useContext(MetaContext);
  if (!value) throw new Error("useMeta must be used inside MetaProvider");
  return value;
}

/** The strategy catalogue order, which is also the colour-slot order. */
export function useStrategyNames(): string[] {
  const { strategies } = useMeta();
  return useMemo(() => strategies.map((s) => s.name), [strategies]);
}
