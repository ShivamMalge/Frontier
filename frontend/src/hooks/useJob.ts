/**
 * Submit a job, then poll it to completion.
 *
 * The pipeline and the backtest both run as jobs because `keras_lstm` takes
 * minutes per ticker; the browser therefore has to survive a run that outlives
 * any reasonable request timeout. Polling stops on a terminal state, on unmount,
 * and on cancellation -- never on its own after N tries, because a legitimate
 * run really can take an hour.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api } from "../api/client";
import type { JobAccepted, JobResult, JobState, JobStatus } from "../api/schema";

const POLL_MS = 1200;

const TERMINAL: JobState[] = ["succeeded", "failed", "cancelled"];

export interface JobView<T> {
  id: string | null;
  state: JobState | null;
  progress: number;
  message: string | null;
  startedAt: string | null;
  finishedAt: string | null;
  result: T | null;
  error: ApiError | null;
  /** True from submission until the job reaches a terminal state. */
  busy: boolean;
}

const IDLE: JobView<never> = {
  id: null,
  state: null,
  progress: 0,
  message: null,
  startedAt: null,
  finishedAt: null,
  result: null,
  error: null,
  busy: false,
};

export function useJob<T>(pick: (result: JobResult["result"]) => T | null) {
  const [view, setView] = useState<JobView<T>>(IDLE as JobView<T>);
  const timer = useRef<number | null>(null);
  const live = useRef(true);

  const stopPolling = useCallback(() => {
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
      timer.current = null;
    }
  }, []);

  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
      stopPolling();
    };
  }, [stopPolling]);

  const finish = useCallback(
    async (id: string, status: JobStatus) => {
      if (status.state !== "succeeded") {
        setView((v) => ({
          ...v,
          state: status.state,
          busy: false,
          message: status.message ?? null,
          finishedAt: status.finished_at ?? null,
          error:
            status.state === "failed"
              ? new ApiError(500, "job_failed", status.message ?? "The job failed.")
              : null,
        }));
        return;
      }
      try {
        const full = await api.jobResult(id);
        if (!live.current) return;
        setView((v) => ({
          ...v,
          state: "succeeded",
          progress: 1,
          busy: false,
          message: full.message ?? null,
          finishedAt: full.finished_at ?? null,
          result: pick(full.result ?? null),
        }));
      } catch (error) {
        if (!live.current) return;
        setView((v) => ({
          ...v,
          busy: false,
          error: error instanceof ApiError ? error : new ApiError(0, "unexpected", String(error)),
        }));
      }
    },
    [pick],
  );

  const poll = useCallback(
    async (id: string) => {
      try {
        const status = await api.jobStatus(id);
        if (!live.current) return;
        setView((v) => ({
          ...v,
          state: status.state,
          progress: status.progress ?? 0,
          message: status.message ?? null,
          startedAt: status.started_at ?? null,
        }));
        if (TERMINAL.includes(status.state)) {
          await finish(id, status);
          return;
        }
        timer.current = window.setTimeout(() => void poll(id), POLL_MS);
      } catch (error) {
        if (!live.current) return;
        setView((v) => ({
          ...v,
          busy: false,
          error: error instanceof ApiError ? error : new ApiError(0, "unexpected", String(error)),
        }));
      }
    },
    [finish],
  );

  const submit = useCallback(
    async (start: () => Promise<JobAccepted>) => {
      stopPolling();
      setView({ ...(IDLE as JobView<T>), busy: true, state: "queued" });
      try {
        const accepted = await start();
        if (!live.current) return;
        setView((v) => ({ ...v, id: accepted.job_id, state: accepted.state }));
        void poll(accepted.job_id);
      } catch (error) {
        if (!live.current) return;
        setView((v) => ({
          ...v,
          busy: false,
          error: error instanceof ApiError ? error : new ApiError(0, "unexpected", String(error)),
        }));
      }
    },
    [poll, stopPolling],
  );

  const cancel = useCallback(async () => {
    if (!view.id) return;
    try {
      await api.cancelJob(view.id);
    } catch {
      // The poll reports the real outcome; a failed cancel is not worth its own
      // error state. On the memory backend a running job can only be flagged.
    }
  }, [view.id]);

  return { ...view, submit, cancel };
}
