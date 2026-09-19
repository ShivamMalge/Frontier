import { describe, expect, it } from "vitest";
import { lastValue, row, series, values, weightsFor } from "./frame";
import type { Frame } from "./schema";

/** Two days, two columns, with the nulls the API uses for NaN and Infinity. */
const frame: Frame = {
  index: ["2023-01-03", "2023-01-04", "2023-01-05"],
  columns: ["AAPL", "MSFT"],
  data: [
    [1.0, 2.0],
    [null, 2.5],
    [1.2, null],
  ],
};

describe("series", () => {
  it("pairs each index label with that column's value", () => {
    expect(series(frame, "AAPL")).toEqual([
      ["2023-01-03", 1.0],
      ["2023-01-04", null],
      ["2023-01-05", 1.2],
    ]);
  });

  it("returns nothing for a column that is not there", () => {
    expect(series(frame, "NOPE")).toEqual([]);
  });

  it("survives a missing frame, which is the pre-run state of every page", () => {
    expect(series(null, "AAPL")).toEqual([]);
    expect(values(undefined, "AAPL")).toEqual([]);
  });
});

describe("lastValue", () => {
  it("skips trailing nulls rather than reporting one", () => {
    expect(lastValue(frame, "MSFT")).toBe(2.5);
  });

  it("is null when the column is entirely null", () => {
    const empty: Frame = { index: ["a"], columns: ["X"], data: [[null]] };
    expect(lastValue(empty, "X")).toBeNull();
  });
});

describe("row", () => {
  it("reads one index label across every column", () => {
    expect(row(frame, "2023-01-04")).toEqual({ AAPL: null, MSFT: 2.5 });
  });
});

describe("weightsFor", () => {
  const weights: Frame = {
    index: ["AAPL", "MSFT", "JPM"],
    columns: ["HRP", "GMV"],
    data: [
      [0.2, 0.5],
      [0.5, 0.2],
      [0.3, 0.3],
    ],
  };

  it("sorts descending so the chart reads top-down", () => {
    expect(weightsFor(weights, "HRP")).toEqual([
      ["MSFT", 0.5],
      ["JPM", 0.3],
      ["AAPL", 0.2],
    ]);
  });

  it("treats a null weight as zero rather than dropping the ticker", () => {
    const ragged: Frame = { index: ["A", "B"], columns: ["X"], data: [[null], [0.4]] };
    expect(weightsFor(ragged, "X")).toEqual([
      ["B", 0.4],
      ["A", 0],
    ]);
  });
});
