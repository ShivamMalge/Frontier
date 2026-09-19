/**
 * Option builders, one per chart form used in the app.
 *
 * Each takes data plus a palette and returns a complete ECharts option. Pages
 * choose a form; they never assemble axes or tooltips themselves.
 */
import type { EChartsCoreOption } from "echarts/core";
import { axis, base, type Palette } from "./theme";
import type { Point } from "../api/frame";
import { num, pct } from "../lib/format";

export interface LineSeries {
  name: string;
  points: Point[];
  /** Index into the categorical ramp. Fixed per entity, never per rank. */
  slot: number;
  dashed?: boolean;
}

/**
 * Time on x, one line per series, crosshair tooltip.
 *
 * 2px lines, no point markers below 120 points (they turn a trend into a
 * caterpillar), and a legend whenever there is more than one series.
 */
export function timeSeries(
  palette: Palette,
  series: LineSeries[],
  options: { format?: (v: number) => string; zoom?: boolean; label: string } = { label: "" },
): EChartsCoreOption {
  const format = options.format ?? ((v: number) => num(v, 2));
  return {
    ...base(palette),
    legend: { ...base(palette).legend, show: series.length > 1 },
    grid: { ...base(palette).grid, top: series.length > 1 ? 34 : 16 },
    tooltip: {
      ...base(palette).tooltip,
      trigger: "axis",
      axisPointer: {
        type: "line",
        lineStyle: { color: palette.muted, width: 1, type: "dashed" },
        label: { backgroundColor: palette.surface, color: palette.text, borderColor: palette.border },
      },
      valueFormatter: (value: unknown) => (typeof value === "number" ? format(value) : "--"),
    },
    xAxis: { type: "time", ...axis(palette) },
    yAxis: { type: "value", scale: true, ...axis(palette, { format }) },
    ...(options.zoom
      ? {
          dataZoom: [
            { type: "inside", throttle: 50 },
            {
              type: "slider",
              height: 18,
              bottom: 6,
              borderColor: palette.border,
              fillerColor: `${palette.accent}22`,
              handleStyle: { color: palette.accent },
              textStyle: { color: palette.muted, fontSize: 10 },
            },
          ],
        }
      : {}),
    series: series.map((line) => ({
      name: line.name,
      type: "line",
      showSymbol: line.points.length < 120,
      symbolSize: 8,
      smooth: false,
      connectNulls: false,
      lineStyle: {
        width: 2,
        color: palette.series[line.slot],
        ...(line.dashed ? { type: "dashed" as const } : {}),
      },
      itemStyle: { color: palette.series[line.slot], borderColor: palette.surface, borderWidth: 2 },
      emphasis: { focus: "series" as const },
      data: line.points,
    })),
  };
}

/**
 * Magnitude by category: horizontal bars, sorted, with the value written at the
 * end of each bar. Direct labels are what makes this readable without a legend
 * -- and what discharges the light-mode contrast relief rule.
 */
export function bars(
  palette: Palette,
  rows: [string, number][],
  options: {
    slot?: number;
    format?: (v: number) => string;
    reference?: { value: number; label: string };
    label: string;
    signed?: boolean;
  },
): EChartsCoreOption {
  const format = options.format ?? ((v: number) => pct(v, 1));
  const slot = options.slot ?? 0;
  return {
    ...base(palette),
    legend: { show: false },
    // The reference line's label sits above the plot, so make room for it.
    grid: { left: 96, right: 62, top: options.reference ? 30 : 10, bottom: 28 },
    tooltip: {
      ...base(palette).tooltip,
      trigger: "item",
      formatter: (p: { name: string; value: number }) => `${p.name}<br/><b>${format(p.value)}</b>`,
    },
    xAxis: { type: "value", ...axis(palette, { format }) },
    yAxis: {
      type: "category",
      // ECharts draws category 0 at the bottom, so reverse to read top-down.
      data: rows.map(([name]) => name).reverse(),
      ...axis(palette),
      splitLine: { show: false },
    },
    series: [
      {
        type: "bar",
        // 4px rounded data-end, square at the baseline.
        itemStyle: {
          color: (p: { value: number }) =>
            options.signed
              ? p.value >= 0
                ? palette.good
                : palette.bad
              : palette.series[slot],
          borderRadius: [0, 4, 4, 0],
        },
        barMaxWidth: 16,
        label: {
          show: true,
          position: "right",
          color: palette.textSecondary,
          fontSize: 11,
          formatter: (p: { value: number }) => format(p.value),
        },
        data: rows.map(([, value]) => value).reverse(),
        ...(options.reference
          ? {
              markLine: {
                silent: true,
                symbol: "none",
                label: {
                  formatter: options.reference.label,
                  color: palette.muted,
                  fontSize: 11,
                  position: "end",
                },
                lineStyle: { color: palette.muted, type: "dashed", width: 1 },
                data: [{ xAxis: options.reference.value }],
              },
            }
          : {}),
      },
    ],
  };
}

export interface FrontierPoint {
  volatility: number;
  expected_return: number;
  sharpe: number;
}

/**
 * Risk on x, return on y: the frontier as a line, the tangency portfolio as a
 * labelled marker. One series, so no legend -- the title names it.
 */
export function frontier(
  palette: Palette,
  points: FrontierPoint[],
  tangency: number,
  extras: { name: string; volatility: number; expected_return: number }[] = [],
): EChartsCoreOption {
  const best = points[tangency];
  return {
    ...base(palette),
    legend: { show: false },
    grid: { left: 64, right: 28, top: 20, bottom: 52 },
    tooltip: {
      ...base(palette).tooltip,
      trigger: "item",
      formatter: (p: { data: [number, number, number, string] }) => {
        const [vol, ret, sharpe, name] = p.data;
        return `${name ?? "frontier"}<br/>return <b>${pct(ret)}</b><br/>vol <b>${pct(vol)}</b><br/>sharpe <b>${num(sharpe)}</b>`;
      },
    },
    xAxis: { type: "value", scale: true, ...axis(palette, { name: "annualised volatility", format: (v) => pct(v, 0) }) },
    yAxis: { type: "value", scale: true, ...axis(palette, { name: "expected return", format: (v) => pct(v, 0) }) },
    series: [
      {
        type: "line",
        showSymbol: true,
        symbolSize: 8,
        lineStyle: { width: 2, color: palette.series[0] },
        itemStyle: { color: palette.series[0], borderColor: palette.surface, borderWidth: 2 },
        data: points.map((p) => [p.volatility, p.expected_return, p.sharpe, "frontier"]),
        ...(best
          ? {
              markPoint: {
                symbol: "circle",
                symbolSize: 13,
                itemStyle: { color: palette.series[1], borderColor: palette.surface, borderWidth: 2 },
                label: {
                  show: true,
                  position: "top",
                  distance: 8,
                  color: palette.text,
                  fontSize: 11,
                  formatter: "max Sharpe",
                },
                data: [{ coord: [best.volatility, best.expected_return] }],
              },
            }
          : {}),
      },
      {
        type: "scatter",
        symbolSize: 11,
        itemStyle: { color: palette.series[2], borderColor: palette.surface, borderWidth: 2 },
        label: {
          show: extras.length <= 4,
          position: "right",
          color: palette.textSecondary,
          fontSize: 11,
          formatter: (p: { data: [number, number, number, string] }) => p.data[3],
        },
        // Marked strategies cluster tightly on a real frontier; drop a label
        // rather than print it on top of its neighbour. The tooltip and the
        // performance table still carry the identity.
        labelLayout: { hideOverlap: true },
        data: extras.map((e) => [e.volatility, e.expected_return, 0, e.name]),
      },
    ],
  };
}
