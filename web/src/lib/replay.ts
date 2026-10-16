// No-camera Replay: bundled clips (mouth-region frame strip + landmarks) that
// run through exactly the same feature code and model as a live capture.
import type { TimedFrame } from '../ml/capture';
import { N_BLENDSHAPES, N_STORED } from '../ml/landmarks';

export interface ReplayClip {
  id: string;
  word: string;
  category: string;
  speaker: string;
  fps: number;
  /** Frames [start, end) that the model reads; the rest is context. */
  modelRange: [number, number];
  frames: number;
  columns: number;
  frameWidth: number;
  frameHeight: number;
  spriteUrl: string;
  landmarksUrl: string;
  /** "grid" for GRID corpus clips, "synthetic" for drawn placeholders. */
  source: 'grid' | 'synthetic';
  credit: string | null;
}

export interface ClipLandmarks {
  frameSize: [number, number];
  crop: [number, number, number, number];
  frames: TimedFrame[];
}

type Json = Record<string, unknown>;
const isObj = (x: unknown): x is Json => typeof x === 'object' && x !== null && !Array.isArray(x);
const num = (x: unknown, what: string): number => {
  if (typeof x !== 'number' || !Number.isFinite(x)) throw new Error(`invalid replay ${what}`);
  return x;
};
const str = (x: unknown, what: string): string => {
  if (typeof x !== 'string') throw new Error(`invalid replay ${what}`);
  return x;
};
// File names come from our own index; still refuse anything that is not a plain name.
const SAFE_NAME = /^[\w.-]+$/;
const fileName = (x: unknown, what: string): string => {
  const s = str(x, what);
  if (!SAFE_NAME.test(s)) throw new Error(`invalid replay ${what}`);
  return s;
};

export function parseReplayIndex(x: unknown, base: string): ReplayClip[] {
  if (!isObj(x) || !Array.isArray(x['clips'])) throw new Error('invalid replay index');
  const source = x['source'] === 'grid' ? 'grid' : 'synthetic';
  const credit = typeof x['credit'] === 'string' ? x['credit'] : null;
  return x['clips'].map((c): ReplayClip => {
    if (!isObj(c) || !Array.isArray(c['modelRange'])) throw new Error('invalid replay clip');
    const frames = num(c['frames'], 'frames');
    const range: [number, number] = [
      num(c['modelRange'][0], 'range'),
      num(c['modelRange'][1], 'range'),
    ];
    if (!(range[0] >= 0 && range[1] <= frames && range[1] > range[0])) {
      throw new Error('invalid replay range');
    }
    return {
      id: fileName(c['id'], 'id'),
      word: str(c['word'], 'word'),
      category: str(c['category'], 'category'),
      speaker: str(c['speaker'], 'speaker'),
      fps: num(c['fps'], 'fps'),
      modelRange: range,
      frames,
      columns: num(c['columns'], 'columns'),
      frameWidth: num(c['frameWidth'], 'frame width'),
      frameHeight: num(c['frameHeight'], 'frame height'),
      spriteUrl: `${base}replay/${fileName(c['sprite'], 'sprite')}`,
      landmarksUrl: `${base}replay/${fileName(c['landmarks'], 'landmarks')}`,
      source,
      credit,
    };
  });
}

function rowsToFloat(rows: unknown, size: number, what: string): (Float32Array | null)[] {
  if (!Array.isArray(rows)) throw new Error(`invalid replay ${what}`);
  return rows.map((row) => {
    if (!Array.isArray(row) || row.length !== size) throw new Error(`invalid replay ${what}`);
    if (row.some((v) => typeof v !== 'number')) return null; // no face in this frame
    return Float32Array.from(row as number[]);
  });
}

export function parseClipLandmarks(x: unknown, fps: number): ClipLandmarks {
  if (!isObj(x) || !Array.isArray(x['frameSize']) || !Array.isArray(x['crop'])) {
    throw new Error('invalid replay landmarks');
  }
  const points = rowsToFloat(x['points'], N_STORED * 3, 'points');
  const blend = rowsToFloat(x['blend'], N_BLENDSHAPES, 'blendshapes');
  const matrix = rowsToFloat(x['matrix'], 16, 'matrices');
  if (blend.length !== points.length || matrix.length !== points.length) {
    throw new Error('invalid replay landmarks: lengths differ');
  }
  const frames = points.map((p, i): TimedFrame => {
    const b = blend[i];
    const m = matrix[i];
    return { t: (i * 1000) / fps, frame: p && b && m ? { points: p, blend: b, matrix: m } : null };
  });
  const [w, h] = x['frameSize'] as unknown[];
  const [cx, cy, cw, ch] = x['crop'] as unknown[];
  return {
    frameSize: [num(w, 'frame size'), num(h, 'frame size')],
    crop: [num(cx, 'crop'), num(cy, 'crop'), num(cw, 'crop'), num(ch, 'crop')],
    frames,
  };
}

async function getJson(url: string): Promise<unknown> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

export async function loadReplayClips(
  base: string = import.meta.env.BASE_URL,
): Promise<ReplayClip[]> {
  return parseReplayIndex(await getJson(`${base}replay/index.json`), base);
}

export async function loadClipLandmarks(clip: ReplayClip): Promise<ClipLandmarks> {
  return parseClipLandmarks(await getJson(clip.landmarksUrl), clip.fps);
}
