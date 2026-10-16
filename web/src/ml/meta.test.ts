import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { availableCategories, isPreview, openingMode, parseLabels, parseMeta } from './meta';

const read = (name: string): unknown =>
  JSON.parse(
    readFileSync(fileURLToPath(new URL(`../../public/models/${name}`, import.meta.url)), 'utf8'),
  );

describe('model files shipped in public/models', () => {
  const meta = parseMeta(read('model-meta.json'));
  const labels = parseLabels(read('labels.json'), meta);

  it('describe a 45-word model over 157 features at 25 fps', () => {
    expect(meta.classes).toHaveLength(45);
    expect(meta.featureDim).toBe(157);
    expect(meta.fps).toBe(25);
    expect(labels.categories.map((c) => c.id)).toEqual([
      'command',
      'colour',
      'preposition',
      'letter',
      'digit',
      'adverb',
    ]);
  });

  it('every category word is a model class', () => {
    for (const c of labels.categories) for (const w of c.words) expect(meta.classes).toContain(w);
  });
});

describe('gating', () => {
  const base = {
    version: 1,
    featureVersion: 1,
    fps: 25,
    featureDim: 157,
    minFrames: 5,
    maxFrames: 50,
    classes: ['bin', 'lay', 'blue'],
    trainedOn: null,
    metrics: null,
    thresholds: { protoMargin: 0.05, protoMinSimilarity: 0.2 },
  };
  const labelsJson = (shipped: boolean[]) => ({
    classes: ['bin', 'lay', 'blue'],
    categories: [
      {
        id: 'command',
        words: ['bin', 'lay'],
        shipped: shipped[0],
        top1: 0.5,
        top3: 1,
        chance1: 0.5,
        threshold: 0.6,
      },
      {
        id: 'colour',
        words: ['blue'],
        shipped: shipped[1],
        top1: 0.2,
        top3: 1,
        chance1: 1,
        threshold: 0.6,
      },
    ],
    fewshotWords: [],
  });
  const trained = {
    ...base,
    trainedOn: 'GRID',
    metrics: { overallTop1: 0.4, fewshotAccuracy: 0.6, fewshotWords: [], testSpeakers: [] },
  };

  it('the untrained preview offers every category and opens on GRID words', () => {
    const meta = parseMeta(base);
    const labels = parseLabels(labelsJson([false, false]), meta);
    expect(isPreview(meta)).toBe(true);
    expect(availableCategories(meta, labels)).toHaveLength(2);
    expect(openingMode(meta, labels)).toBe('grid');
  });

  it('a trained model offers only categories that passed the gate', () => {
    const meta = parseMeta(trained);
    const labels = parseLabels(labelsJson([true, false]), meta);
    expect(availableCategories(meta, labels).map((c) => c.id)).toEqual(['command']);
    expect(openingMode(meta, labels)).toBe('grid');
  });

  it('opens on Teach when no category passed', () => {
    const meta = parseMeta(trained);
    const labels = parseLabels(labelsJson([false, false]), meta);
    expect(openingMode(meta, labels)).toBe('teach');
  });

  it('rejects files that do not match the feature contract', () => {
    expect(() => parseMeta({ ...base, featureDim: 100 })).toThrow(/feature contract/);
    expect(() => parseMeta({ ...base, featureVersion: 2 })).toThrow(/version/);
    expect(() =>
      parseLabels({ ...labelsJson([true, true]), classes: ['x'] }, parseMeta(base)),
    ).toThrow(/classes differ/);
  });
});
