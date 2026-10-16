import { describe, expect, it } from 'vitest';
import {
  categoryGuesses,
  cosine,
  nearestPrototype,
  prototypeOf,
  softmax,
  type TaughtWord,
} from './decide';

const unit = (xs: number[]) => {
  const n = Math.hypot(...xs);
  return Float32Array.from(xs, (x) => x / n);
};

describe('softmax', () => {
  it('sums to one and keeps order', () => {
    const p = softmax([1, 3, 2]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(p[1]).toBeGreaterThan(p[2]!);
  });
});

describe('categoryGuesses', () => {
  const classes = ['bin', 'lay', 'place', 'blue', 'green', 'red'];

  it('ranks only the chosen category and is sure above its threshold', () => {
    const logits = Float32Array.from([0, 0, 0, 1, 9, 2]);
    const result = categoryGuesses(logits, classes, ['blue', 'green', 'red'], 0.5);
    expect(result.guesses.map((g) => g.word)).toEqual(['green', 'red', 'blue']);
    expect(result.sure).toBe(true);
    expect(result.guesses[0]!.p).toBeGreaterThan(0.9);
  });

  it('is not sure below the calibrated threshold', () => {
    const logits = Float32Array.from([0, 0, 0, 1, 1.2, 1.1]);
    const result = categoryGuesses(logits, classes, ['blue', 'green', 'red'], 0.5);
    expect(result.sure).toBe(false);
    expect(result.guesses).toHaveLength(3);
  });

  it('returns at most three guesses', () => {
    const logits = Float32Array.from([1, 2, 3, 4, 5, 6]);
    expect(categoryGuesses(logits, classes, classes, 0).guesses).toHaveLength(3);
  });

  it('ignores words the model does not know', () => {
    const logits = Float32Array.from([1, 2, 3, 4, 5, 6]);
    const result = categoryGuesses(logits, classes, ['bin', 'set'], 0.5);
    expect(result.guesses.map((g) => g.word)).toEqual(['bin']);
  });
});

describe('prototypes', () => {
  it('cosine of unit vectors', () => {
    expect(cosine(unit([1, 0]), unit([0, 1]))).toBeCloseTo(0, 6);
    expect(cosine(unit([1, 1]), unit([1, 1]))).toBeCloseTo(1, 6);
  });

  it('prototype is the normalised mean of the takes', () => {
    const p = prototypeOf([unit([1, 0]), unit([0, 1])]);
    expect(p[0]).toBeCloseTo(Math.SQRT1_2, 6);
    expect(Math.hypot(...p)).toBeCloseTo(1, 6);
  });

  const words: TaughtWord[] = [
    { label: 'play', takes: [unit([1, 0, 0]), unit([0.9, 0.1, 0])] },
    { label: 'pause', takes: [unit([0, 1, 0])] },
    { label: 'next', takes: [unit([0, 0, 1])] },
  ];
  const thresholds = { protoMargin: 0.1, protoMinSimilarity: 0.5 };

  it('finds the nearest taught word', () => {
    const match = nearestPrototype(unit([1, 0.05, 0]), words, thresholds);
    expect(match.best?.label).toBe('play');
    expect(match.sure).toBe(true);
    expect(match.ranked.map((r) => r.label)).toEqual(['play', 'pause', 'next']);
  });

  it('is not sure when two words are nearly tied', () => {
    const match = nearestPrototype(unit([1, 1, 0]), words, thresholds);
    expect(match.sure).toBe(false);
  });

  it('is not sure when nothing is similar enough', () => {
    const match = nearestPrototype(unit([-1, -1, -1]), words, thresholds);
    expect(match.sure).toBe(false);
  });

  it('needs at least two words to judge a margin', () => {
    const match = nearestPrototype(unit([1, 0, 0]), words.slice(0, 1), thresholds);
    expect(match.sure).toBe(false);
    expect(match.best?.label).toBe('play');
  });

  it('skips words without takes', () => {
    const match = nearestPrototype(
      unit([0, 0, 1]),
      [...words, { label: 'empty', takes: [] }],
      thresholds,
    );
    expect(match.ranked.map((r) => r.label)).not.toContain('empty');
  });
});
