// Validation of the JSON files the pipeline writes into public/.
import { FEATURE_DIM } from './features';

export interface CategoryInfo {
  id: string;
  words: string[];
  shipped: boolean;
  top1: number | null;
  top3: number | null;
  chance1: number;
  threshold: number;
}

export interface ModelMeta {
  version: number;
  featureVersion: number;
  fps: number;
  featureDim: number;
  minFrames: number;
  maxFrames: number;
  classes: string[];
  trainedOn: string | null;
  metrics: {
    overallTop1: number;
    fewshotAccuracy: number;
    fewshotWords: string[];
    testSpeakers: string[];
  } | null;
  thresholds: { protoMargin: number; protoMinSimilarity: number };
}

export interface Labels {
  classes: string[];
  categories: CategoryInfo[];
  fewshotWords: string[];
}

type Json = Record<string, unknown>;

const isObj = (x: unknown): x is Json => typeof x === 'object' && x !== null && !Array.isArray(x);
const isNum = (x: unknown): x is number => typeof x === 'number' && Number.isFinite(x);
const isStrArray = (x: unknown): x is string[] =>
  Array.isArray(x) && x.every((v) => typeof v === 'string');

function fail(what: string): never {
  throw new Error(`invalid ${what}`);
}

export const FEATURE_VERSION = 1;

export function parseMeta(x: unknown): ModelMeta {
  if (!isObj(x)) fail('model-meta.json');
  const { version, featureVersion, fps, featureDim, minFrames, maxFrames, classes } = x;
  if (!isNum(version) || featureVersion !== FEATURE_VERSION) fail('model-meta.json version');
  if (!isNum(fps) || featureDim !== FEATURE_DIM) fail('model-meta.json feature contract');
  if (!isNum(minFrames) || !isNum(maxFrames) || minFrames < 1 || maxFrames < minFrames) {
    fail('model-meta.json frame limits');
  }
  if (!isStrArray(classes) || classes.length === 0) fail('model-meta.json classes');
  const t = x['thresholds'];
  if (!isObj(t) || !isNum(t['protoMargin']) || !isNum(t['protoMinSimilarity'])) {
    fail('model-meta.json thresholds');
  }
  const m = x['metrics'];
  let metrics: ModelMeta['metrics'] = null;
  if (m !== null) {
    if (!isObj(m) || !isNum(m['overallTop1']) || !isNum(m['fewshotAccuracy'])) {
      fail('model-meta.json metrics');
    }
    metrics = {
      overallTop1: m['overallTop1'],
      fewshotAccuracy: m['fewshotAccuracy'],
      fewshotWords: isStrArray(m['fewshotWords']) ? m['fewshotWords'] : [],
      testSpeakers: isStrArray(m['testSpeakers']) ? m['testSpeakers'] : [],
    };
  }
  const trainedOn = x['trainedOn'];
  return {
    version,
    featureVersion,
    fps,
    featureDim,
    minFrames,
    maxFrames,
    classes,
    trainedOn: typeof trainedOn === 'string' ? trainedOn : null,
    metrics,
    thresholds: { protoMargin: t['protoMargin'], protoMinSimilarity: t['protoMinSimilarity'] },
  };
}

const numOrNull = (x: unknown): number | null => (isNum(x) ? x : null);

export function parseLabels(x: unknown, meta: ModelMeta): Labels {
  if (!isObj(x) || !isStrArray(x['classes']) || !Array.isArray(x['categories'])) {
    fail('labels.json');
  }
  if (x['classes'].join() !== meta.classes.join()) fail('labels.json: classes differ from model');
  const categories = x['categories'].map((c): CategoryInfo => {
    if (!isObj(c) || typeof c['id'] !== 'string' || !isStrArray(c['words'])) {
      fail('labels.json category');
    }
    if (!isNum(c['threshold']) || !isNum(c['chance1'])) fail('labels.json category numbers');
    return {
      id: c['id'],
      words: c['words'].filter((w) => meta.classes.includes(w)),
      shipped: c['shipped'] === true,
      top1: numOrNull(c['top1']),
      top3: numOrNull(c['top3']),
      chance1: c['chance1'],
      threshold: c['threshold'],
    };
  });
  return {
    classes: x['classes'],
    categories,
    fewshotWords: isStrArray(x['fewshotWords']) ? x['fewshotWords'] : [],
  };
}

/** True until a trained model with held-out metrics replaces the stub. */
export function isPreview(meta: ModelMeta): boolean {
  return meta.metrics === null;
}

/**
 * Categories offered in "GRID words". A trained model offers only categories
 * that passed the held-out gate; the untrained preview offers all of them.
 */
export function availableCategories(meta: ModelMeta, labels: Labels): CategoryInfo[] {
  return isPreview(meta) ? labels.categories : labels.categories.filter((c) => c.shipped);
}

export type Mode = 'grid' | 'teach' | 'remote';

/** GRID words if any category is offered, otherwise Teach your own words. */
export function openingMode(meta: ModelMeta, labels: Labels): Mode {
  return availableCategories(meta, labels).length > 0 ? 'grid' : 'teach';
}

export const CATEGORY_LABELS: Record<string, string> = {
  command: 'Commands',
  colour: 'Colours',
  preposition: 'Prepositions',
  letter: 'Letters',
  digit: 'Digits',
  adverb: 'Adverbs',
};
