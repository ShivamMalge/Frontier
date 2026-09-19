import { describe, expect, it } from "vitest";
import { MAX_SERIES, makeSlots } from "./theme";

describe("makeSlots", () => {
  it("gives each entity its own slot", () => {
    const slot = makeSlots();
    const active = ["HRP", "GMV", "RiskParity"];
    expect(active.map((name) => slot(name, active))).toEqual([0, 1, 2]);
  });

  it("keeps a colour with its entity when another is removed", () => {
    // The rule this enforces: filtering must not repaint the survivors.
    const slot = makeSlots();
    const all = ["HRP", "GMV", "RiskParity"];
    for (const name of all) slot(name, all);

    const fewer = ["HRP", "RiskParity"];
    expect(slot("HRP", fewer)).toBe(0);
    expect(slot("RiskParity", fewer)).toBe(2);
  });

  it("reuses a slot only once its holder has left the selection", () => {
    const slot = makeSlots();
    const first = ["HRP", "GMV"];
    for (const name of first) slot(name, first);

    const second = ["HRP", "MinCVaR"];
    expect(slot("MinCVaR", second)).toBe(1);
  });

  it("never invents a ninth hue", () => {
    const slot = makeSlots();
    const many = Array.from({ length: 12 }, (_, i) => `S${i}`);
    const slots = many.map((name) => slot(name, many));
    expect(Math.max(...slots)).toBeLessThan(MAX_SERIES);
  });
});
