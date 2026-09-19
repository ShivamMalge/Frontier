/** Progress, elapsed time and cancellation for a running job. */
import { duration } from "../lib/format";
import { Note } from "./Bits";

export function JobBar({
  state,
  progress,
  message,
  startedAt,
  finishedAt,
  onCancel,
}: {
  state: string | null;
  progress: number;
  message: string | null;
  startedAt: string | null;
  finishedAt: string | null;
  onCancel?: () => void;
}) {
  if (!state) return null;
  if (state === "failed") return <Note kind="error">{message ?? "The job failed."}</Note>;
  if (state === "cancelled") return <Note kind="warn">Cancelled.</Note>;
  if (state === "succeeded") {
    return (
      <div className="status-line">
        <span>
          <span className="dot ok" />
          finished in {duration(startedAt, finishedAt)}
        </span>
      </div>
    );
  }
  return (
    <div style={{ display: "grid", gap: 8 }}>
      <div className="status-line" style={{ justifyContent: "space-between" }}>
        <span>
          <span className="spinner" style={{ marginRight: 8 }} />
          {message ?? state} · {duration(startedAt)}
        </span>
        {onCancel && (
          <button type="button" className="ghost" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>
      <div className="progress">
        <div style={{ width: `${Math.round(progress * 100)}%` }} />
      </div>
    </div>
  );
}
