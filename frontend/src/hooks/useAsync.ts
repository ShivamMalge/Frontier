/**
 * Request state for one-shot calls.
 *
 * Every in-flight request is abortable and the result of a stale one is
 * dropped: switching pages mid-forecast must not paint the old answer over the
 * new page.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";

export interface AsyncState<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
}

export function useAsync<T>() {
  const [state, setState] = useState<AsyncState<T>>({ data: null, error: null, loading: false });
  const generation = useRef(0);
  const controller = useRef<AbortController | null>(null);

  const run = useCallback(async (work: (signal: AbortSignal) => Promise<T>) => {
    controller.current?.abort();
    const mine = ++generation.current;
    const abort = new AbortController();
    controller.current = abort;
    setState((previous) => ({ ...previous, loading: true, error: null }));
    try {
      const data = await work(abort.signal);
      if (mine === generation.current) setState({ data, error: null, loading: false });
      return data;
    } catch (error) {
      if (abort.signal.aborted || mine !== generation.current) return null;
      const failure =
        error instanceof ApiError ? error : new ApiError(0, "unexpected", String(error));
      setState({ data: null, error: failure, loading: false });
      return null;
    }
  }, []);

  const reset = useCallback(() => {
    controller.current?.abort();
    generation.current += 1;
    setState({ data: null, error: null, loading: false });
  }, []);

  useEffect(() => () => controller.current?.abort(), []);

  return { ...state, run, reset };
}

/** Fetch once on mount. For the metadata endpoints, which never change. */
export function useOnce<T>(work: (signal: AbortSignal) => Promise<T>, deps: unknown[] = []) {
  const state = useAsync<T>();
  const { run } = state;
  useEffect(() => {
    void run(work);
    // The caller owns the dependency list; `work` is re-created every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}
