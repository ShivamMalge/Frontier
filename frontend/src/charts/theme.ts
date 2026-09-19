/**
 * The bridge between the stylesheet and ECharts.
 *
 * Colours are never hard-coded in an option object: they are read back from the
 * CSS custom properties in styles.css, so light and dark are defined in exactly
 * one place and a chart can never disagree with the surrounding page.
 */
import { useEffect, useState } from "react";

export interface Palette {
  surface: string;
  border: string;
  text: string;
  textSecondary: string;
  muted: string;
  accent: string;
  good: string;
  bad: string;
  /** The eight categorical slots, in fixed order. Never cycled. */
  series: string[];
}

/** The most series any chart here may draw. Past this, fold or facet. */
export const MAX_SERIES = 8;

function readVar(styles: CSSStyleDeclaration, name: string): string {
  return styles.getPropertyValue(name).trim();
}

export function readPalette(): Palette {
  const styles = getComputedStyle(document.documentElement);
  return {
    surface: readVar(styles, "--surface-1"),
    border: readVar(styles, "--border"),
    text: readVar(styles, "--text-primary"),
    textSecondary: readVar(styles, "--text-secondary"),
    muted: readVar(styles, "--text-muted"),
    accent: readVar(styles, "--accent"),
    good: readVar(styles, "--good"),
    bad: readVar(styles, "--bad"),
    series: Array.from({ length: MAX_SERIES }, (_, i) => readVar(styles, `--series-${i + 1}`)),
  };
}

/** Re-reads the palette when the OS colour scheme flips. */
export function usePalette(): Palette {
  const [palette, setPalette] = useState<Palette>(() => readPalette());
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setPalette(readPalette());
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  return palette;
}

/**
 * Assigns colour slots to entities and keeps them.
 *
 * Deselecting one series must not repaint the others, so a colour belongs to a
 * name for as long as that name is on screen -- not to its position in the
 * current selection.
 */
export function makeSlots(): (name: string, active: string[]) => number {
  const held = new Map<string, number>();
  return (name, active) => {
    const existing = held.get(name);
    if (existing !== undefined && active.includes(name)) return existing;
    const taken = new Set(active.map((other) => held.get(other)).filter((s) => s !== undefined));
    for (let slot = 0; slot < MAX_SERIES; slot += 1) {
      if (!taken.has(slot)) {
        held.set(name, slot);
        return slot;
      }
    }
    return MAX_SERIES - 1;
  };
}

/** Grid, axes and tooltip shell every chart shares. */
export function base(palette: Palette) {
  return {
    backgroundColor: "transparent",
    animationDuration: 260,
    textStyle: {
      fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
      fontSize: 12,
    },
    grid: { left: 58, right: 20, top: 30, bottom: 44, containLabel: false },
    tooltip: {
      backgroundColor: palette.surface,
      borderColor: palette.border,
      borderWidth: 1,
      padding: [8, 10],
      textStyle: { color: palette.text, fontSize: 12 },
      extraCssText: "box-shadow: 0 4px 14px rgba(0,0,0,0.12); border-radius: 6px;",
    },
    legend: {
      type: "scroll" as const,
      top: 0,
      left: 0,
      icon: "roundRect",
      itemWidth: 10,
      itemHeight: 10,
      itemGap: 14,
      // Legend text wears text tokens, not the series colour: the swatch beside
      // it already carries identity.
      textStyle: { color: palette.textSecondary, fontSize: 12 },
      inactiveColor: palette.muted,
    },
  };
}

/** Recessive axis: a hairline, muted labels, horizontal split lines only. */
export function axis(
  palette: Palette,
  options: { name?: string; format?: (v: number) => string } = {},
) {
  return {
    nameTextStyle: { color: palette.muted, fontSize: 11 },
    ...(options.name ? { name: options.name, nameLocation: "middle" as const, nameGap: 32 } : {}),
    axisLine: { lineStyle: { color: palette.border } },
    axisTick: { show: false },
    axisLabel: {
      color: palette.textSecondary,
      fontSize: 11,
      ...(options.format ? { formatter: options.format } : {}),
    },
    splitLine: {
      lineStyle: {
        color: palette.border,
        type: "dashed" as const,
        opacity: 0.7,
      },
    },
  };
}

/**
 * Height class for a bar chart of `rows` bars.
 *
 * A fixed height crams eighteen tickers into the space six need, and the labels
 * collide. Ten strategies or eighteen tickers are both realistic here.
 */
export function barChartClass(rows: number): string {
  if (rows <= 6) return "chart short";
  if (rows <= 12) return "chart";
  return "chart tall";
}
