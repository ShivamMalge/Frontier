/**
 * A sized, disposable ECharts canvas.
 *
 * Only the chart types actually used are imported: pulling `echarts` whole adds
 * roughly a megabyte of chart types this app never draws.
 */

import { BarChart, LineChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  MarkPointComponent,
  TooltipComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useRef } from "react";

echarts.use([
  BarChart,
  LineChart,
  ScatterChart,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  MarkPointComponent,
  TooltipComponent,
  CanvasRenderer,
]);

interface Props {
  option: echarts.EChartsCoreOption;
  /** Applied to the container; use the .chart size classes. */
  className?: string;
  /** What the chart shows, for screen readers. The table beside it is the real
      accessible view -- this is the label, not a substitute. */
  label: string;
}

export function Chart({ option, className = "chart", label }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!host.current) return;
    chart.current = echarts.init(host.current, undefined, {
      renderer: "canvas",
    });
    const observer = new ResizeObserver(() => chart.current?.resize());
    observer.observe(host.current);
    return () => {
      observer.disconnect();
      chart.current?.dispose();
      chart.current = null;
    };
  }, []);

  // `true` replaces the option rather than merging: a series that disappears
  // from the data must disappear from the chart, not linger from the last render.
  useEffect(() => {
    chart.current?.setOption(option, true);
  }, [option]);

  return <div ref={host} className={className} role="img" aria-label={label} />;
}
