/**
 * The only place that talks to the API.
 *
 * Paths are relative: the Vite dev server and the nginx image both proxy /api
 * and /health to the service, so the browser only ever sees one origin and CORS
 * never enters the picture.
 */
import type {
  BackendListResponse,
  CoverageResponse,
  ForecastRequest,
  ForecastResponse,
  FrontierRequest,
  FrontierResponse,
  HealthResponse,
  IngestRequest,
  JobAccepted,
  JobResult,
  JobStatus,
  OptimizeRequest,
  OptimizeResponse,
  PerformanceRequest,
  PerformanceResponse,
  PipelineRequest,
  PricesRequest,
  PricesResponse,
  QueryRequest,
  QueryResponse,
  ReturnsResponse,
  StrategyListResponse,
  TrackedRunList,
  TrackingStatus,
  BacktestRequest,
} from "./schema";

const BASE = import.meta.env.VITE_API_BASE ?? "";

/** An error the API reported, carrying its machine-readable code. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly context: Record<string, string> = {},
  ) {
    super(message);
    this.name = "ApiError";
  }
}

type ErrorBody = {
  error?: { code?: string; message?: string; context?: Record<string, string> };
  detail?: unknown;
};

/** Turn every failure shape the service can emit into one ApiError. */
async function toError(response: Response): Promise<ApiError> {
  let body: ErrorBody = {};
  try {
    body = (await response.json()) as ErrorBody;
  } catch {
    return new ApiError(response.status, "unreadable", response.statusText);
  }
  if (body.error?.message) {
    return new ApiError(
      response.status,
      body.error.code ?? "service_error",
      body.error.message,
      body.error.context ?? {},
    );
  }
  if (Array.isArray(body.detail)) {
    // FastAPI's own 422: report the field, since that is what the user must fix.
    const parts = body.detail.map((item) => {
      const d = item as { loc?: unknown[]; msg?: string };
      const field = (d.loc ?? []).slice(1).join(".");
      return field ? `${field}: ${d.msg}` : (d.msg ?? "invalid");
    });
    return new ApiError(response.status, "validation_error", parts.join("; "));
  }
  return new ApiError(response.status, "http_error", String(body.detail ?? response.statusText));
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch (cause) {
    throw new ApiError(0, "unreachable", "The API is not reachable. Is it running?", {
      cause: String(cause),
    });
  }
  if (!response.ok) throw await toError(response);
  return (await response.json()) as T;
}

const get = <T,>(path: string, signal?: AbortSignal) => request<T>(path, { signal });
const post = <T,>(path: string, body: unknown, signal?: AbortSignal) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body), signal });

const V1 = "/api/v1";

export const api = {
  health: (signal?: AbortSignal) => get<HealthResponse>("/health", signal),
  universe: (signal?: AbortSignal) => get<import("./schema").UniverseResponse>(`${V1}/meta/universe`, signal),
  strategies: (signal?: AbortSignal) => get<StrategyListResponse>(`${V1}/meta/strategies`, signal),
  backends: (signal?: AbortSignal) => get<BackendListResponse>(`${V1}/meta/backends`, signal),

  prices: (body: PricesRequest, signal?: AbortSignal) =>
    post<PricesResponse>(`${V1}/market/prices`, body, signal),
  returns: (body: PricesRequest, signal?: AbortSignal) =>
    post<ReturnsResponse>(`${V1}/market/returns`, body, signal),

  forecast: (body: ForecastRequest, signal?: AbortSignal) =>
    post<ForecastResponse>(`${V1}/forecast`, body, signal),

  optimize: (body: OptimizeRequest, signal?: AbortSignal) =>
    post<OptimizeResponse>(`${V1}/portfolio/optimize`, body, signal),
  performance: (body: PerformanceRequest, signal?: AbortSignal) =>
    post<PerformanceResponse>(`${V1}/portfolio/performance`, body, signal),
  frontier: (body: FrontierRequest, signal?: AbortSignal) =>
    post<FrontierResponse>(`${V1}/portfolio/frontier`, body, signal),

  submitPipeline: (body: PipelineRequest) => post<JobAccepted>(`${V1}/pipeline/runs`, body),
  submitBacktest: (body: BacktestRequest) => post<JobAccepted>(`${V1}/backtest/runs`, body),
  submitIngest: (body: IngestRequest) => post<JobAccepted>(`${V1}/data/ingest`, body),

  jobStatus: (id: string, signal?: AbortSignal) =>
    get<JobStatus>(`${V1}/pipeline/runs/${id}`, signal),
  jobResult: (id: string, signal?: AbortSignal) =>
    get<JobResult>(`${V1}/pipeline/runs/${id}/result`, signal),
  cancelJob: (id: string) => request<JobStatus>(`${V1}/pipeline/runs/${id}`, { method: "DELETE" }),

  coverage: (signal?: AbortSignal) => get<CoverageResponse>(`${V1}/data/coverage`, signal),
  query: (body: QueryRequest, signal?: AbortSignal) =>
    post<QueryResponse>(`${V1}/data/query`, body, signal),

  trackingStatus: (signal?: AbortSignal) => get<TrackingStatus>(`${V1}/tracking/status`, signal),
  trackingRuns: (limit: number, signal?: AbortSignal) =>
    get<TrackedRunList>(`${V1}/tracking/runs?limit=${limit}`, signal),
};
