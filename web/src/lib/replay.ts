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
const int = (x: unknown, what: string, min: number, max: number): number => {
  if (!Number.isInteger(x) || (x as number) < min || (x as number) > max) {
    throw new Error(`invalid replay ${what}`);
  }
  return x as number;
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

// The index and clips are this site's own files; the limits below are generous
// bounds that keep a broken file from driving a huge canvas or a long loop.
const MAX_CLIPS = 24;
const MAX_CLIP_FRAMES = 250; // 10 s at 25 fps
const MAX_FRAME_SIDE = 1024;
const MAX_FPS = 120;

export function parseReplayIndex(x: unknown, base: string): ReplayClip[] {
  if (!isObj(x) || !Array.isArray(x['clips']) || x['clips'].length > MAX_CLIPS) {
    throw new Error('invalid replay index');
  }
  const source = x['source'] === 'grid' ? 'grid' : 'synthetic';
  const credit = typeof x['credit'] === 'string' ? x['credit'] : null;
  return x['clips'].map((c): ReplayClip => {
    if (!isObj(c) || !Array.isArray(c['modelRange'])) throw new Error('invalid replay clip');
    const frames = int(c['frames'], 'frames', 1, MAX_CLIP_FRAMES);
    const range: [number, number] = [
      int(c['modelRange'][0], 'range', 0, frames),
      int(c['modelRange'][1], 'range', 0, frames),
    ];
    if (range[1] <= range[0]) throw new Error('invalid replay range');
    const fps = num(c['fps'], 'fps');
    if (fps <= 0 || fps > MAX_FPS) throw new Error('invalid replay fps');
    return {
      id: fileName(c['id'], 'id'),
      word: str(c['word'], 'word'),
      category: str(c['category'], 'category'),
      speaker: str(c['speaker'], 'speaker'),
      fps,
      modelRange: range,
      frames,
      columns: int(c['columns'], 'columns', 1, frames),
      frameWidth: int(c['frameWidth'], 'frame width', 1, MAX_FRAME_SIDE),
      frameHeight: int(c['frameHeight'], 'frame height', 1, MAX_FRAME_SIDE),
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
    if (row.some((v) => v === null)) return null; // no face in this frame
    if (!row.every((v) => typeof v === 'number' && Number.isFinite(v))) {
      throw new Error(`invalid replay ${what}`);
    }
    return Float32Array.from(row as number[]);
  });
}

const positive = (x: unknown, what: string): number => {
  const v = num(x, what);
  if (v <= 0) throw new Error(`invalid replay ${what}`);
  return v;
};

/** A clip's landmark file; `frames`, when given, is the count the index promised. */
export function parseClipLandmarks(x: unknown, fps: number, frames?: number): ClipLandmarks {
  if (
    !isObj(x) ||
    !Array.isArray(x['frameSize']) ||
    x['frameSize'].length !== 2 ||
    !Array.isArray(x['crop']) ||
    x['crop'].length !== 4 ||
    !Array.isArray(x['points']) ||
    x['points'].length > MAX_CLIP_FRAMES ||
    (frames !== undefined && x['points'].length !== frames)
  ) {
    throw new Error('invalid replay landmarks');
  }
  const points = rowsToFloat(x['points'], N_STORED * 3, 'points');
  const blend = rowsToFloat(x['blend'], N_BLENDSHAPES, 'blendshapes');
  const matrix = rowsToFloat(x['matrix'], 16, 'matrices');
  if (blend.length !== points.length || matrix.length !== points.length) {
    throw new Error('invalid replay landmarks: lengths differ');
  }
  const timed = points.map((p, i): TimedFrame => {
    const b = blend[i];
    const m = matrix[i];
    return { t: (i * 1000) / fps, frame: p && b && m ? { points: p, blend: b, matrix: m } : null };
  });
  const [w, h] = x['frameSize'] as unknown[];
  const [cx, cy, cw, ch] = x['crop'] as unknown[];
  return {
    frameSize: [positive(w, 'frame size'), positive(h, 'frame size')],
    crop: [num(cx, 'crop'), num(cy, 'crop'), positive(cw, 'crop'), positive(ch, 'crop')],
    frames: timed,
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
  return parseClipLandmarks(await getJson(clip.landmarksUrl), clip.fps, clip.frames);
}

/** Throws unless a decoded frame strip holds every frame the clip says it has. */
export function checkSprite(
  clip: Pick<ReplayClip, 'frames' | 'columns' | 'frameWidth' | 'frameHeight'>,
  width: number,
  height: number,
): void {
  const rows = Math.ceil(clip.frames / clip.columns);
  if (width < clip.columns * clip.frameWidth || height < rows * clip.frameHeight) {
    throw new Error(`replay sprite is ${width}x${height}, too small for its ${clip.frames} frames`);
  }
}
