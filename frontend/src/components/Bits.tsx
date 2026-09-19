/** Small presentational pieces shared by every page. */
import type { ReactNode } from "react";

export function Card({
  title,
  hint,
  actions,
  children,
}: {
  title?: string;
  hint?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="card">
      {(title || actions) && (
        <header>
          <div>
            {title && <h2>{title}</h2>}
            {hint && <p>{hint}</p>}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="stat">
      <dt>{label}</dt>
      <dd>
        {value} {sub && <small>{sub}</small>}
      </dd>
    </div>
  );
}

export function Note({
  kind = "info",
  children,
}: {
  kind?: "info" | "warn" | "error";
  children: ReactNode;
}) {
  const mark = { info: "note", warn: "warning", error: "error" }[kind];
  return (
    <div className={`note ${kind}`}>
      <b>{mark}</b>
      <span>{children}</span>
    </div>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    // biome-ignore lint/a11y/noLabelWithoutControl: the control is the caller's `children`, nested inside this label -- the association the rule wants, just not one it can see statically
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <span className="hint">{hint}</span>}
    </label>
  );
}

/** Multi-select as toggle chips, with an optional colour swatch per entry. */
export function Chips({
  options,
  selected,
  onToggle,
  colorOf,
  disabledWhen,
}: {
  options: string[];
  selected: string[];
  onToggle: (value: string) => void;
  colorOf?: (value: string) => string | undefined;
  disabledWhen?: (value: string) => boolean;
}) {
  return (
    <div className="chips">
      {options.map((option) => {
        const on = selected.includes(option);
        const color = on ? colorOf?.(option) : undefined;
        return (
          <button
            key={option}
            type="button"
            className="chip"
            aria-pressed={on}
            disabled={!on && disabledWhen?.(option)}
            onClick={() => onToggle(option)}
          >
            {color && <span className="swatch" style={{ background: color }} />}
            {option}
          </button>
        );
      })}
    </div>
  );
}

export function Spinner({ children }: { children?: ReactNode }) {
  return (
    <span className="status-line">
      <span className="spinner" /> {children}
    </span>
  );
}
