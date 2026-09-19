/**
 * Helpers for the API's column-oriented `Frame`.
 *
 * A Frame is `{ index, columns, data }` where `data[row][column]`, with null
 * standing in for NaN and Infinity -- neither of which is valid JSON. Charts
 * want `[x, y]` pairs per series, so the transposition happens once, here,
 * rather than in every page.
 */
import type { Frame } from "./schema";

export type Point = [string, number | null];

/** Column names in the order the API returned them. */
export function columns(frame: Frame | null | undefined): string[] {
  return frame?.columns ?? [];
}

/** One column as `[indexLabel, value]` pairs, ready for a time series. */
export function series(frame: Frame | null | undefined, column: string): Point[] {
  if (!frame) return [];
  const at = frame.columns.indexOf(column);
  if (at < 0) return [];
  return frame.index.map((label, row) => [label, valueAt(frame, row, at)]);
}

/** One column's values, without the index. */
export function values(frame: Frame | null | undefined, column: string): (number | null)[] {
  return series(frame, column).map(([, value]) => value);
}

/** The last non-null value of a column -- the end of a growth curve, say. */
export function lastValue(frame: Frame | null | undefined, column: string): number | null {
  const column_values = values(frame, column);
  for (let i = column_values.length - 1; i >= 0; i -= 1) {
    const value = column_values[i];
    if (value != null) return value;
  }
  return null;
}

/** One row as a `{ column: value }` record. */
export function row(frame: Frame | null | undefined, label: string): Record<string, number | null> {
  const out: Record<string, number | null> = {};
  if (!frame) return out;
  const at = frame.index.indexOf(label);
  if (at < 0) return out;
  frame.columns.forEach((column, col) => {
    out[column] = valueAt(frame, at, col);
  });
  return out;
}

/** A single cell, normalising undefined (a ragged row) to null. */
function valueAt(frame: Frame, rowIndex: number, columnIndex: number): number | null {
  const values_row = frame.data[rowIndex] as (number | null)[] | undefined;
  const value = values_row?.[columnIndex];
  return value == null ? null : value;
}

/** Turn a column of weights into descending `[ticker, weight]` pairs. */
export function weightsFor(frame: Frame | null | undefined, strategy: string): [string, number][] {
  if (!frame) return [];
  const at = frame.columns.indexOf(strategy);
  if (at < 0) return [];
  return frame.index
    .map((ticker, rowIndex): [string, number] => [ticker, valueAt(frame, rowIndex, at) ?? 0])
    .sort((a, b) => b[1] - a[1]);
}
