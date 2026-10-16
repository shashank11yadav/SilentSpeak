// Auto-trimming a held capture to the part where the lips actually move.
import { GEOMETRY_DIM, PRESENCE } from './features';

export type TrimResult =
  { ok: true; start: number; end: number } | { ok: false; reason: 'too-short' | 'no-motion' };

export interface TrimOptions {
  /** Frames kept either side of the motion (training used +-2 around each word). */
  pad: number;
  minFrames: number;
}

/** Lip-shape change between consecutive frames (0 when either has no face). */
export function motionEnergy(frames: readonly Float32Array[]): number[] {
  return frames.map((f, t) => {
    const prev = frames[t - 1];
    if (!prev || f[PRESENCE] !== 1 || prev[PRESENCE] !== 1) return 0;
    let sum = 0;
    for (let i = 0; i < GEOMETRY_DIM; i++) sum += Math.abs((f[i] ?? 0) - (prev[i] ?? 0));
    return sum;
  });
}

function smooth(xs: number[]): number[] {
  return xs.map((_, i) => {
    let sum = 0;
    let n = 0;
    for (let j = i - 1; j <= i + 1; j++) {
      const v = xs[j];
      if (v !== undefined) {
        sum += v;
        n++;
      }
    }
    return sum / n;
  });
}

function percentile(xs: number[], q: number): number {
  const sorted = [...xs].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] ?? 0;
}

// Below this summed change per frame the lips are treated as still.
const MIN_PEAK_ENERGY = 0.02;
const THRESHOLD_FRACTION = 0.25;

/** Frame range [start, end) around the lip motion, or why there is none. */
export function trimByMotion(frames: readonly Float32Array[], opts: TrimOptions): TrimResult {
  if (frames.length < opts.minFrames) return { ok: false, reason: 'too-short' };
  const raw = motionEnergy(frames);
  // The level is set on smoothed energy so one jittery frame cannot set the
  // peak; the boundaries come from the raw energy so they stay frame-exact.
  const smoothed = smooth(raw);
  const peak = Math.max(...smoothed);
  if (peak < MIN_PEAK_ENERGY) return { ok: false, reason: 'no-motion' };
  const base = percentile(smoothed, 0.2);
  const threshold = base + THRESHOLD_FRACTION * (peak - base);
  const first = raw.findIndex((e) => e >= threshold);
  const last = raw.length - 1 - [...raw].reverse().findIndex((e) => e >= threshold);
  // energy[t] measures the change from frame t-1 to t, so motion starts at t-1.
  const start = Math.max(0, first - 1 - opts.pad);
  const end = Math.min(frames.length, last + 1 + opts.pad);
  if (end - start < opts.minFrames) return { ok: false, reason: 'too-short' };
  return { ok: true, start, end };
}

/** The `size`-frame window inside [start, end) with the most motion. */
export function bestWindow(
  energy: readonly number[],
  start: number,
  end: number,
  size: number,
): [number, number] {
  if (end - start <= size) return [start, end];
  let best = start;
  let bestSum = -1;
  for (let s = start; s + size <= end; s++) {
    let sum = 0;
    for (let i = s; i < s + size; i++) sum += energy[i] ?? 0;
    if (sum > bestSum) {
      bestSum = sum;
      best = s;
    }
  }
  return [best, best + size];
}
