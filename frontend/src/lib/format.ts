/** Number and date formatting. Charts and tables must agree, so it lives here. */

/** A ratio as a percentage: 0.0731 -> "7.31%". */
export function pct(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "--";
  return `${(value * 100).toFixed(digits)}%`;
}

/** A plain number, fixed to `digits`. */
export function num(value: number | null | undefined, digits = 3): string {
  if (value == null || !Number.isFinite(value)) return "--";
  return value.toFixed(digits);
}

/** Thousands-separated integer. */
export function count(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "--";
  return value.toLocaleString("en-US");
}

/** Seconds elapsed between two API timestamps, as "1m 04s". */
export function duration(from?: string | null, to?: string | null): string {
  if (!from) return "--";
  const start = new Date(from).getTime();
  const end = to ? new Date(to).getTime() : Date.now();
  const seconds = Math.max(0, Math.round((end - start) / 1000));
  const minutes = Math.floor(seconds / 60);
  return minutes ? `${minutes}m ${String(seconds % 60).padStart(2, "0")}s` : `${seconds}s`;
}

/** An ISO timestamp as a local, sortable-looking string. */
export function timestamp(value?: string | null): string {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
