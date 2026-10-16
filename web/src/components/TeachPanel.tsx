import { Plus, Trash2, X } from 'lucide-react';
import { useState, type SubmitEvent } from 'react';
import type { PrototypeMatch, TaughtWord } from '../ml/decide';
import { MAX_TAKES, MAX_WORDS, normaliseLabel } from '../lib/word-store';
import type { Setup } from '../state';
import { Guesses } from './Guesses';
import { ResultHeadline } from './ResultHeadline';

interface Props {
  words: TaughtWord[];
  setup: Setup | null;
  match: PrototypeMatch | null;
  persistent: boolean;
  onStartSetup: (labels: string[], takesPerWord: number) => void;
  onCancelSetup: () => void;
  onDelete: (label: string) => void;
  onClear: () => void;
}

const QUICK = ['play', 'pause', 'next'];
const TAKES = 3;

export function TeachPanel(props: Props) {
  const { words, setup, match } = props;
  if (setup) return <SetupProgress setup={setup} onCancel={props.onCancelSetup} />;
  if (words.length === 0)
    return <QuickSetup onStart={props.onStartSetup} persistent={props.persistent} />;
  return (
    <div className="space-y-5">
      <div className="rounded-xl border border-line bg-paper/60 p-4" data-testid="teach-result">
        {words.length < 2 ? (
          <p className="text-sm text-ink-muted">Teach at least two words, then say one of them.</p>
        ) : match?.best ? (
          <>
            <ResultHeadline
              word={match.best.label}
              sure={match.sure}
              hint="Too close between two words. Say it again, or add a take."
            />
            <Guesses
              caption="Closest taught words"
              dim={!match.sure}
              bars={match.ranked.slice(0, 3).map((r) => ({
                label: r.label,
                value: Math.max(0, r.similarity),
                display: r.similarity.toFixed(2),
              }))}
            />
          </>
        ) : (
          <p className="text-sm text-ink-muted">
            Say one of your words. The closest one shows here.
          </p>
        )}
      </div>

      <div>
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium">
            Your words{' '}
            <span className="text-ink-muted">
              ({words.length} of {MAX_WORDS})
            </span>
          </h3>
          <button
            type="button"
            onClick={props.onClear}
            className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-danger hover:bg-danger-soft"
          >
            <Trash2 className="size-3.5" aria-hidden="true" /> Clear all
          </button>
        </div>
        <ul className="mt-2 flex flex-wrap gap-2">
          {words.map((w) => (
            <li
              key={w.label}
              className="flex items-center gap-1 rounded-full border border-line bg-surface py-1 pr-1 pl-3 text-sm"
            >
              <span className="font-medium">{w.label}</span>
              <span className="text-xs text-ink-muted">
                {w.takes.length} take{w.takes.length === 1 ? '' : 's'}
              </span>
              {w.takes.length < MAX_TAKES && (
                <button
                  type="button"
                  aria-label={`Record another take of ${w.label}`}
                  onClick={() => {
                    props.onStartSetup([w.label], 1);
                  }}
                  className="grid size-6 place-items-center rounded-full hover:bg-surface-muted"
                >
                  <Plus className="size-3.5" aria-hidden="true" />
                </button>
              )}
              <button
                type="button"
                aria-label={`Forget ${w.label}`}
                onClick={() => {
                  props.onDelete(w.label);
                }}
                className="grid size-6 place-items-center rounded-full hover:bg-surface-muted"
              >
                <X className="size-3.5" aria-hidden="true" />
              </button>
            </li>
          ))}
        </ul>
        {words.length < MAX_WORDS && <AddWord words={words} onAdd={props.onStartSetup} />}
        <p className="mt-3 text-xs text-ink-muted">
          {props.persistent
            ? 'Saved in this browser only, as numbers describing lip motion. No video is kept.'
            : 'This browser blocks site storage, so your words last until you close the tab.'}
        </p>
      </div>
    </div>
  );
}

function SetupProgress({ setup, onCancel }: { setup: Setup; onCancel: () => void }) {
  const total = setup.labels.length * setup.takesPerWord;
  const done = setup.index * setup.takesPerWord + setup.take;
  const word = setup.labels[setup.index] ?? '';
  return (
    <div className="space-y-4" data-testid="setup">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium">
          Teaching · take {done + 1} of {total}
        </span>
        <button
          type="button"
          onClick={onCancel}
          className="rounded-md px-2 py-1 text-ink-muted hover:bg-surface-muted"
        >
          Cancel
        </button>
      </div>
      <div
        className="h-2 overflow-hidden rounded-full bg-surface-muted"
        role="progressbar"
        aria-label="Setup progress"
        aria-valuemin={0}
        aria-valuemax={total}
        aria-valuenow={done}
      >
        <div
          className="h-full rounded-full bg-accent transition-all"
          style={{ width: `${(done / total) * 100}%` }}
        />
      </div>
      <div
        className="rounded-xl border border-accent/40 bg-accent-soft p-5 text-center"
        aria-live="polite"
      >
        <p className="text-sm text-ink-muted">Hold the button and silently mouth</p>
        <p className="mt-1 text-4xl font-semibold tracking-tight">{word}</p>
        <div className="mt-3 flex justify-center gap-1.5" aria-hidden="true">
          {Array.from({ length: setup.takesPerWord }, (_, i) => (
            <span
              key={i}
              className={`size-2.5 rounded-full ${i < setup.take ? 'bg-accent' : 'bg-ink-muted/30'}`}
            />
          ))}
        </div>
        <p className="mt-2 text-xs text-ink-muted">
          Take {setup.take + 1} of {setup.takesPerWord} for this word
        </p>
      </div>
    </div>
  );
}

function QuickSetup({
  onStart,
  persistent,
}: {
  onStart: (labels: string[], takes: number) => void;
  persistent: boolean;
}) {
  const [labels, setLabels] = useState(QUICK);
  const submit = (e: SubmitEvent<HTMLFormElement>) => {
    e.preventDefault();
    onStart(labels.map(normaliseLabel).filter(Boolean), TAKES);
  };
  return (
    <form onSubmit={submit} className="space-y-4">
      <div>
        <h3 className="text-base font-semibold">Teach it your own words</h3>
        <p className="mt-1 text-sm text-ink-muted">
          Record each word three times. Recognition then compares new attempts with your takes, so
          it works for your face and your way of speaking. About a minute.
        </p>
      </div>
      <div className="grid grid-cols-3 gap-2">
        {labels.map((l, i) => (
          <label key={i} className="text-xs font-medium text-ink-muted">
            Word {i + 1}
            <input
              value={l}
              maxLength={24}
              onChange={(e) => {
                setLabels(labels.map((x, j) => (j === i ? e.target.value : x)));
              }}
              className="mt-1 w-full rounded-lg border border-line bg-paper px-2.5 py-2 text-sm text-ink"
            />
          </label>
        ))}
      </div>
      <button
        type="submit"
        className="w-full rounded-xl bg-ink px-4 py-2.5 text-sm font-semibold text-paper hover:opacity-90"
      >
        Start quick setup · 3 words × 3 takes
      </button>
      {!persistent && (
        <p className="text-xs text-ink-muted">
          This browser blocks site storage, so taught words last until you close the tab.
        </p>
      )}
    </form>
  );
}

function AddWord({
  words,
  onAdd,
}: {
  words: TaughtWord[];
  onAdd: (labels: string[], takes: number) => void;
}) {
  const [value, setValue] = useState('');
  const label = normaliseLabel(value);
  const exists = words.some((w) => w.label === label);
  return (
    <form
      className="mt-3 flex gap-2"
      onSubmit={(e) => {
        e.preventDefault();
        if (!label || exists) return;
        onAdd([label], TAKES);
        setValue('');
      }}
    >
      <label className="sr-only" htmlFor="new-word">
        New word
      </label>
      <input
        id="new-word"
        value={value}
        maxLength={24}
        placeholder="Add a word, e.g. back"
        onChange={(e) => {
          setValue(e.target.value);
        }}
        className="min-w-0 flex-1 rounded-lg border border-line bg-paper px-3 py-2 text-sm"
      />
      <button
        type="submit"
        disabled={!label || exists}
        className="rounded-lg border border-line px-3 py-2 text-sm font-medium hover:bg-surface-muted disabled:opacity-50"
      >
        Teach
      </button>
    </form>
  );
}
