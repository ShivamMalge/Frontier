/**
 * The table that sits beside every chart.
 *
 * It is not decoration: three light-mode series colours fall below 3:1 contrast
 * against the surface, and the rule for that is relief -- a readable table view
 * or direct labels. Every chart in this app ships one or both.
 */
import type { ReactNode } from "react";

export interface Column<T> {
  key: string;
  header: string;
  /** Right-aligned, tabular figures. Use for anything numeric. */
  num?: boolean;
  render: (row: T) => ReactNode;
  /** Sort value; omit to make the column unsortable. */
  sort?: (row: T) => number | string;
}

interface Props<T> {
  rows: T[];
  columns: Column<T>[];
  rowKey: (row: T) => string;
  /** Highlights one row, e.g. the strategy the pipeline selected. */
  highlight?: (row: T) => boolean;
  onSelect?: (row: T) => void;
  empty?: string;
}

export function Table<T>({ rows, columns, rowKey, highlight, onSelect, empty }: Props<T>) {
  if (!rows.length) return <div className="empty">{empty ?? "Nothing to show yet."}</div>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key} className={column.num ? "num" : undefined} scope="col">
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={rowKey(row)}
              className={highlight?.(row) ? "selected" : undefined}
              onClick={onSelect ? () => onSelect(row) : undefined}
              style={onSelect ? { cursor: "pointer" } : undefined}
            >
              {columns.map((column) => (
                <td key={column.key} className={column.num ? "num" : undefined}>
                  {column.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** A number coloured by sign, for returns and drawdowns. */
export function Signed({ value, render }: { value: number | null; render: (v: number) => string }) {
  if (value == null || !Number.isFinite(value)) return <span className="mono">--</span>;
  return <span className={value >= 0 ? "pos" : "neg"}>{render(value)}</span>;
}
