/**
 * The universe / date / backend controls, shared by every page that runs work.
 *
 * Backends that the deployment does not have are listed and disabled rather
 * than hidden: /health reports an honest capability set, and so should this.
 */

import type { ForecastBackend } from "../api/schema";
import { useMeta } from "../state";
import { Chips, Field } from "./Bits";

export function RunControls({
  showBackend = true,
  children,
}: {
  showBackend?: boolean;
  children?: React.ReactNode;
}) {
  const { universe, backends, params, setParams, toggleTicker } = useMeta();

  return (
    <div style={{ display: "grid", gap: 14 }}>
      <div>
        <div className="field" style={{ minWidth: 0 }}>
          <label htmlFor="universe-chips">Universe</label>
        </div>
        <div id="universe-chips">
          <Chips
            options={universe?.tickers ?? []}
            selected={params.tickers}
            onToggle={toggleTicker}
          />
        </div>
      </div>

      <div className="controls">
        <Field label="Start">
          <input
            type="date"
            value={params.start}
            onChange={(event) => setParams({ start: event.target.value })}
          />
        </Field>
        <Field label="End">
          <input
            type="date"
            value={params.end}
            onChange={(event) => setParams({ end: event.target.value })}
          />
        </Field>
        {showBackend && (
          <Field
            label="Backend"
            hint={backends.find((b) => b.name === params.backend)?.description}
          >
            <select
              value={params.backend}
              onChange={(event) => setParams({ backend: event.target.value as ForecastBackend })}
            >
              {backends.map((backend) => (
                <option key={backend.name} value={backend.name} disabled={!backend.available}>
                  {backend.name}
                  {backend.available ? "" : ` (needs ${backend.requires})`}
                </option>
              ))}
            </select>
          </Field>
        )}
        {children}
      </div>
    </div>
  );
}
