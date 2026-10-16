// Mirrors pipeline/src/silentspeak/features.py. The shared parity fixture
// (shared/fixtures/features.json) checks both to 1e-4; read the Python module
// for the reasoning behind each step.
import {
  BLEND_PAIRS,
  BLEND_SINGLE,
  BLENDSHAPE_NAMES,
  EYE_OUTER,
  LIP_MIDLINE,
  LIP_PAIRS,
  N_LIPS,
  N_STORED,
  storedRow,
} from './landmarks';

export const GEOMETRY_DIM = 3 * LIP_PAIRS.length + 2 * LIP_MIDLINE.length; // 62
export const BLEND_DIM = BLEND_SINGLE.length + BLEND_PAIRS.length; // 16
export const FRAME_DIM = GEOMETRY_DIM + BLEND_DIM + 1; // 79
export const PRESENCE = FRAME_DIM - 1;
export const FEATURE_DIM = FRAME_DIM + PRESENCE; // 157
export const FPS = 25;

/**
 * One landmarker result: the stored points in image-height units (x, y, z per
 * point, N_STORED points), 52 blendshape scores and the 4x4 facial
 * transformation matrix in row-major order.
 */
export interface Frame {
  points: Float32Array;
  blend: Float32Array;
  matrix: Float32Array;
}

const PAIR_ROWS = LIP_PAIRS.map(([a, b]) => [storedRow(a), storedRow(b)] as const);
const MID_ROWS = LIP_MIDLINE.map((m) => storedRow(m));
const EYE_ROWS = [storedRow(EYE_OUTER[0]), storedRow(EYE_OUTER[1])] as const;
const BLEND_INDEX = new Map<string, number>(BLENDSHAPE_NAMES.map((n, i) => [n, i]));
const blendIndex = (name: string): number => BLEND_INDEX.get(name) ?? -1;
const BLEND_SINGLE_IDX = BLEND_SINGLE.map(blendIndex);
const BLEND_PAIR_IDX = BLEND_PAIRS.map(([a, b]) => [blendIndex(a), blendIndex(b)] as const);
const MIN_EYE_DISTANCE = 1e-6;

/** MediaPipe normalised landmarks (x, y, z triples) -> image-height units. */
export function toHeightUnits(xyz: Float32Array, width: number, height: number): Float32Array {
  const aspect = width / height;
  const out = new Float32Array(xyz.length);
  for (let i = 0; i < xyz.length; i += 3) {
    out[i] = (xyz[i] ?? 0) * aspect;
    out[i + 1] = xyz[i + 1] ?? 0;
    out[i + 2] = (xyz[i + 2] ?? 0) * aspect;
  }
  return out;
}

const at = (a: ArrayLike<number>, i: number): number => a[i] ?? Number.NaN;

/** Features of one frame; all zeros (presence 0) when there is no usable face. */
export function frameFeatures(frame: Frame | null): Float32Array {
  const out = new Float32Array(FRAME_DIM);
  if (!frame || frame.points.length !== N_STORED * 3 || frame.matrix.length !== 16) return out;
  const { points, blend, matrix } = frame;
  for (let i = 0; i < points.length; i++) if (!Number.isFinite(at(points, i))) return out;
  for (let i = 0; i < 16; i++) if (!Number.isFinite(at(matrix, i))) return out;

  // Rotation part of the row-major matrix and its average column length.
  const r = (row: number, col: number) => at(matrix, row * 4 + col);
  let scale = 0;
  for (let c = 0; c < 3; c++) scale += Math.hypot(r(0, c), r(1, c), r(2, c));
  scale /= 3;
  if (!(scale > 0)) return out;

  // q = v @ (R / scale), with v the point flipped to y-up, z-toward-viewer.
  const q = new Float64Array(N_STORED * 3);
  const centre = [0, 0, 0];
  for (let n = 0; n < N_STORED; n++) {
    const vx = at(points, 3 * n);
    const vy = -at(points, 3 * n + 1);
    const vz = -at(points, 3 * n + 2);
    for (let c = 0; c < 3; c++) {
      const value = (vx * r(0, c) + vy * r(1, c) + vz * r(2, c)) / scale;
      q[3 * n + c] = value;
      if (n < N_LIPS) centre[c] = (centre[c] ?? 0) + value;
    }
  }
  for (let c = 0; c < 3; c++) centre[c] = (centre[c] ?? 0) / N_LIPS;
  for (let n = 0; n < N_STORED; n++) {
    for (let c = 0; c < 3; c++) q[3 * n + c] = at(q, 3 * n + c) - (centre[c] ?? 0);
  }
  const [e0, e1] = EYE_ROWS;
  const eye = Math.hypot(
    at(q, 3 * e0) - at(q, 3 * e1),
    at(q, 3 * e0 + 1) - at(q, 3 * e1 + 1),
    at(q, 3 * e0 + 2) - at(q, 3 * e1 + 2),
  );
  if (!(eye >= MIN_EYE_DISTANCE)) return out;
  const p = (row: number, c: number) => at(q, 3 * row + c) / eye;

  let k = 0;
  for (const [ia, ib] of PAIR_ROWS) {
    out[k++] = (p(ia, 0) - p(ib, 0)) / 2;
    out[k++] = (p(ia, 1) + p(ib, 1)) / 2;
    out[k++] = (p(ia, 2) + p(ib, 2)) / 2;
  }
  for (const im of MID_ROWS) {
    out[k++] = p(im, 1);
    out[k++] = p(im, 2);
  }
  const b = (i: number) => {
    const v = blend[i];
    return v === undefined || Number.isNaN(v) ? 0 : v;
  };
  for (const i of BLEND_SINGLE_IDX) out[k++] = b(i);
  for (const [ia, ib] of BLEND_PAIR_IDX) out[k++] = b(ia) + b(ib);
  out[PRESENCE] = 1;
  return out;
}

/** Append frame-to-frame deltas: [T][FRAME_DIM] -> [T][FEATURE_DIM]. */
export function sequenceFeatures(frames: readonly Float32Array[]): Float32Array[] {
  return frames.map((f, t) => {
    const out = new Float32Array(FEATURE_DIM);
    out.set(f.subarray(0, FRAME_DIM));
    const prev = t > 0 ? frames[t - 1] : undefined;
    if (prev) {
      const gate = (f[PRESENCE] ?? 0) * (prev[PRESENCE] ?? 0);
      for (let i = 0; i < PRESENCE; i++) {
        out[FRAME_DIM + i] = ((f[i] ?? 0) - (prev[i] ?? 0)) * gate;
      }
    }
    return out;
  });
}

/**
 * Sample frames at fractional positions: linear between neighbours, except
 * next to a frame with no face (last value 0), where the nearer frame is taken.
 */
export function interpolateAt(seq: readonly Float32Array[], positions: number[]): Float32Array[] {
  const n = seq.length;
  return positions.map((raw) => {
    const pos = Math.min(Math.max(raw, 0), n - 1);
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, n - 1);
    const a = seq[i0] ?? new Float32Array(0);
    const b = seq[i1] ?? a;
    let w = pos - i0;
    const last = a.length - 1;
    if (!(a[last] === 1 && b[last] === 1)) w = w < 0.5 ? 0 : 1;
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = (a[i] ?? 0) * (1 - w) + (b[i] ?? 0) * w;
    return out;
  });
}

/** Resample from a uniform source rate to dstFps. */
export function resample(
  seq: readonly Float32Array[],
  srcFps: number,
  dstFps: number = FPS,
): Float32Array[] {
  if (!(srcFps > 0) || !(dstFps > 0))
    throw new Error(`fps must be positive: ${srcFps} -> ${dstFps}`);
  if (seq.length <= 1) return seq.map((f) => Float32Array.from(f));
  const nOut = Math.floor(((seq.length - 1) * dstFps) / srcFps + 1e-9) + 1;
  const step = srcFps / dstFps;
  return interpolateAt(
    seq,
    Array.from({ length: nOut }, (_, k) => k * step),
  );
}

/** Resample frames captured at the given timestamps (ms) to a uniform dstFps. */
export function resampleTimed(
  seq: readonly Float32Array[],
  timesMs: readonly number[],
  dstFps: number = FPS,
): Float32Array[] {
  if (seq.length !== timesMs.length) throw new Error('one timestamp per frame');
  if (seq.length <= 1) return seq.map((f) => Float32Array.from(f));
  const time = (i: number) => timesMs[i] ?? 0;
  const t0 = time(0);
  const duration = time(timesMs.length - 1) - t0;
  const period = 1000 / dstFps;
  const nOut = Math.floor(duration / period + 1e-9) + 1;
  const positions: number[] = [];
  let j = 0;
  for (let k = 0; k < nOut; k++) {
    const t = t0 + k * period;
    while (j < timesMs.length - 2 && time(j + 1) <= t) j++;
    const ta = time(j);
    const tb = time(j + 1);
    positions.push(tb > ta ? j + Math.min(1, Math.max(0, (t - ta) / (tb - ta))) : j);
  }
  return interpolateAt(seq, positions);
}

/** Flatten [T][FEATURE_DIM] into the model's [1, T, FEATURE_DIM] input buffer. */
export function flatten(seq: readonly Float32Array[]): Float32Array {
  const out = new Float32Array(seq.length * FEATURE_DIM);
  seq.forEach((f, t) => {
    out.set(f.subarray(0, FEATURE_DIM), t * FEATURE_DIM);
  });
  return out;
}
