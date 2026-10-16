// Turning model outputs into answers, including the honest "not sure".

export function softmax(xs: readonly number[]): number[] {
  const max = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export interface Guess {
  word: string;
  p: number;
}

export interface CategoryResult {
  guesses: Guess[];
  /** False when the top probability is under the category's calibrated threshold. */
  sure: boolean;
}

/** Top 3 among one category's words, with probabilities renormalised within it. */
export function categoryGuesses(
  logits: Float32Array,
  classes: readonly string[],
  words: readonly string[],
  threshold: number,
): CategoryResult {
  const idx = words.map((w) => classes.indexOf(w)).filter((i) => i >= 0);
  const probs = softmax(idx.map((i) => logits[i] ?? 0));
  const guesses = idx
    .map((i, k) => ({ word: classes[i] ?? '', p: probs[k] ?? 0 }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 3);
  return { guesses, sure: (guesses[0]?.p ?? 0) >= threshold };
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return na > 0 && nb > 0 ? dot / Math.sqrt(na * nb) : 0;
}

/** L2-normalised mean of several takes' embeddings. */
export function prototypeOf(takes: readonly ArrayLike<number>[]): Float32Array {
  const dim = takes[0]?.length ?? 0;
  const out = new Float32Array(dim);
  for (const t of takes) for (let i = 0; i < dim; i++) out[i] = (out[i] ?? 0) + (t[i] ?? 0);
  const norm = Math.hypot(...out);
  if (norm > 0) for (let i = 0; i < dim; i++) out[i] = (out[i] ?? 0) / norm;
  return out;
}

export interface TaughtWord {
  label: string;
  takes: Float32Array[];
}

export interface PrototypeThresholds {
  protoMargin: number;
  protoMinSimilarity: number;
}

export interface PrototypeMatch {
  best: { label: string; similarity: number } | null;
  ranked: { label: string; similarity: number }[];
  margin: number;
  sure: boolean;
}

/** Cosine nearest prototype, "not sure" unless clearly ahead of the runner-up. */
export function nearestPrototype(
  embedding: ArrayLike<number>,
  words: readonly TaughtWord[],
  thresholds: PrototypeThresholds,
): PrototypeMatch {
  const ranked = words
    .filter((w) => w.takes.length > 0)
    .map((w) => ({ label: w.label, similarity: cosine(embedding, prototypeOf(w.takes)) }))
    .sort((a, b) => b.similarity - a.similarity);
  const best = ranked[0] ?? null;
  const second = ranked[1];
  const margin = best && second ? best.similarity - second.similarity : 0;
  const sure =
    best !== null &&
    second !== undefined &&
    margin >= thresholds.protoMargin &&
    best.similarity >= thresholds.protoMinSimilarity;
  return { best, ranked, margin, sure };
}
