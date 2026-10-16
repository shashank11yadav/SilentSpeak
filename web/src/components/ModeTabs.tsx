import { BookOpen, Grid3x3, Tv } from 'lucide-react';
import { useRef, type KeyboardEvent } from 'react';
import type { Mode } from '../ml/meta';

const TABS: { id: Mode; label: string; Icon: typeof Grid3x3 }[] = [
  { id: 'grid', label: 'GRID words', Icon: Grid3x3 },
  { id: 'teach', label: 'Teach words', Icon: BookOpen },
  { id: 'remote', label: 'Silent remote', Icon: Tv },
];

/** WAI-ARIA tabs: arrow keys move between tabs, the panel follows. */
export function ModeTabs({ mode, onMode }: { mode: Mode; onMode: (m: Mode) => void }) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const onKey = (e: KeyboardEvent, index: number) => {
    const delta = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
    if (!delta && e.key !== 'Home' && e.key !== 'End') return;
    e.preventDefault();
    const next =
      e.key === 'Home'
        ? 0
        : e.key === 'End'
          ? TABS.length - 1
          : (index + delta + TABS.length) % TABS.length;
    const tab = TABS[next];
    if (!tab) return;
    onMode(tab.id);
    refs.current[next]?.focus();
  };
  return (
    <div
      role="tablist"
      aria-label="Mode"
      className="grid grid-cols-3 gap-1 rounded-xl bg-surface-muted p-1"
    >
      {TABS.map(({ id, label, Icon }, i) => {
        const active = id === mode;
        return (
          <button
            key={id}
            ref={(el) => {
              refs.current[i] = el;
            }}
            id={`tab-${id}`}
            type="button"
            role="tab"
            aria-selected={active}
            aria-controls={`panel-${id}`}
            tabIndex={active ? 0 : -1}
            onClick={() => {
              onMode(id);
            }}
            onKeyDown={(e) => {
              onKey(e, i);
            }}
            className={`flex flex-col items-center justify-center gap-1 rounded-lg px-2 py-2 text-xs font-medium transition sm:flex-row sm:gap-2 sm:text-sm ${
              active ? 'bg-surface text-ink shadow-sm' : 'text-ink-muted hover:text-ink'
            }`}
          >
            <Icon className="size-4 shrink-0" aria-hidden="true" />
            <span className="text-center leading-tight">{label}</span>
          </button>
        );
      })}
    </div>
  );
}
