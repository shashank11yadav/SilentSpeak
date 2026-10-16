/** The big answer line: the word, or an honest "Not sure". */
export function ResultHeadline({
  word,
  sure,
  hint,
}: {
  word: string | null;
  sure: boolean;
  hint: string;
}) {
  return (
    <div role="status" aria-live="polite" className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
      {sure && word ? (
        <>
          <span className="text-sm font-medium text-ink-muted">You said</span>
          <span className="text-3xl font-semibold tracking-tight text-accent">{word}</span>
        </>
      ) : (
        <>
          <span className="text-3xl font-semibold tracking-tight">Not sure</span>
          <span className="text-sm text-ink-muted">{hint}</span>
        </>
      )}
    </div>
  );
}
