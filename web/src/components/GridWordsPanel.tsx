import type { CategoryResult } from '../ml/decide';
import { CATEGORY_LABELS, type CategoryInfo } from '../ml/meta';
import { Guesses } from './Guesses';
import { ResultHeadline } from './ResultHeadline';

interface Props {
  categories: CategoryInfo[];
  categoryId: string | null;
  onCategory: (id: string) => void;
  preview: boolean;
  result: { result: CategoryResult; truth: string | null; categoryId: string } | null;
}

const pct = (x: number) => `${Math.round(x * 100)}%`;

export function GridWordsPanel({ categories, categoryId, onCategory, preview, result }: Props) {
  const active = categories.find((c) => c.id === categoryId) ?? categories[0];
  if (!active) {
    return (
      <p className="text-sm text-ink-muted">
        No GRID category reached the accuracy bar on held-out speakers, so this mode is hidden.
        Teach your own words instead.
      </p>
    );
  }
  const shown = result && result.categoryId === active.id ? result : null;
  return (
    <div className="space-y-5">
      <fieldset>
        <legend className="mb-2 text-sm font-medium">Pick what kind of word you will say</legend>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {categories.map((c) => {
            const checked = c.id === active.id;
            return (
              <label
                key={c.id}
                className={`relative cursor-pointer rounded-xl border px-3 py-2.5 transition has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-[var(--focus)] ${
                  checked ? 'border-accent bg-accent-soft' : 'border-line hover:border-ink-muted/50'
                }`}
              >
                <input
                  type="radio"
                  name="category"
                  value={c.id}
                  checked={checked}
                  onChange={() => {
                    onCategory(c.id);
                  }}
                  className="sr-only"
                />
                <span className="block text-sm font-semibold">{CATEGORY_LABELS[c.id] ?? c.id}</span>
                <span className="mt-0.5 flex items-center justify-between gap-2 text-xs text-ink-muted">
                  <span>{c.words.length} words</span>
                  <span className={c.top1 === null ? '' : 'font-semibold text-ink'}>
                    {c.top1 === null ? (preview ? 'untrained' : 'n/a') : pct(c.top1)}
                  </span>
                </span>
              </label>
            );
          })}
        </div>
      </fieldset>

      <div>
        <p className="text-sm font-medium">Say one of these</p>
        <ul
          className="mt-2 flex flex-wrap gap-1.5"
          aria-label={`${CATEGORY_LABELS[active.id] ?? active.id} words`}
        >
          {active.words.map((w) => (
            <li key={w} className="rounded-md bg-surface-muted px-2 py-0.5 font-mono text-sm">
              {w}
            </li>
          ))}
        </ul>
        <p className="mt-2 text-xs text-ink-muted">
          {active.top1 === null
            ? `Accuracy appears here once the model is trained. Chance is ${pct(active.chance1)}.`
            : `On speakers the model never saw, the top guess is right ${pct(active.top1)} of the time (chance ${pct(active.chance1)}).`}
        </p>
      </div>

      <div className="rounded-xl border border-line bg-paper/60 p-4" data-testid="grid-result">
        {shown ? (
          <>
            <ResultHeadline
              word={shown.result.guesses[0]?.word ?? null}
              sure={shown.result.sure}
              hint="Below the confidence bar for this category. Try again, a little slower."
            />
            {shown.truth && (
              <p className="mt-1 text-sm text-ink-muted">
                The clip says <span className="font-semibold text-ink">{shown.truth}</span>
              </p>
            )}
            <Guesses
              caption="Top guesses"
              dim={!shown.result.sure}
              bars={shown.result.guesses.map((g) => ({
                label: g.word,
                value: g.p,
                display: pct(g.p),
              }))}
            />
          </>
        ) : (
          <p className="text-sm text-ink-muted">Your top three guesses will appear here.</p>
        )}
      </div>
    </div>
  );
}
