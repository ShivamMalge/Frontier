import { describe, expect, it } from "vitest";
import { count, duration, num, pct } from "./format";

describe("pct", () => {
  it("scales a ratio into a percentage", () => {
    expect(pct(0.0731)).toBe("7.31%");
    expect(pct(-0.184, 1)).toBe("-18.4%");
  });

  it("shows nulls as an em-dash rather than NaN%", () => {
    // The API sends null for NaN and Infinity, and directional accuracy is
    // genuinely null for the flat baseline.
    expect(pct(null)).toBe("--");
    expect(pct(Number.NaN)).toBe("--");
    expect(pct(Number.POSITIVE_INFINITY)).toBe("--");
  });
});

describe("num", () => {
  it("fixes the digits so columns line up", () => {
    expect(num(1.0112, 3)).toBe("1.011");
    expect(num(35.6741, 2)).toBe("35.67");
  });
});

describe("count", () => {
  it("separates thousands", () => {
    expect(count(2014)).toBe("2,014");
  });
});

describe("duration", () => {
  it("reads minutes and seconds for a long run", () => {
    const start = "2026-01-01T00:00:00Z";
    const end = "2026-01-01T00:01:04Z";
    expect(duration(start, end)).toBe("1m 04s");
  });

  it("reads plain seconds under a minute", () => {
    expect(duration("2026-01-01T00:00:00Z", "2026-01-01T00:00:12Z")).toBe("12s");
  });
});
