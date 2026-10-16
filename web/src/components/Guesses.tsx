export interface Bar {
  label: string;
  /** 0..1 for the bar length. */
  value: number;
  /** What to print next to the bar, e.g. "82%" or "0.91". */
  display: string;
}

/** A ranked list of guesses with proportional bars. */
export function Guesses({ bars, dim, caption }: { bars: Bar[]; dim: boolean; caption: string }) {
  return (
    <ol aria-label={caption} className="mt-3 space-y-2">
      {bars.map((b, i) => (
        <li
          key={b.label}
          className="grid grid-cols-[minmax(4.5rem,auto)_1fr_3.5rem] items-center gap-3"
        >
          <span
            className={`truncate font-medium ${i === 0 && !dim ? 'text-ink' : 'text-ink-muted'}`}
          >
            {b.label}
          </span>
          <span className="h-2.5 overflow-hidden rounded-full bg-surface-muted" aria-hidden="true">
            <span
              className={`block h-full rounded-full ${i === 0 && !dim ? 'bg-accent' : 'bg-ink-muted/40'}`}
              style={{ width: `${Math.max(2, Math.round(b.value * 100))}%` }}
            />
          </span>
          <span className="text-right font-mono text-sm tabular-nums text-ink-muted">
            {b.display}
          </span>
        </li>
      ))}
    </ol>
  );
}
