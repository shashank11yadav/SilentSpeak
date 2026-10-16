import { describe, expect, it } from 'vitest';
import { FRAME_DIM, PRESENCE } from './features';
import { bestWindow, motionEnergy, trimByMotion } from './trim';

/** A still mouth, then `moving` frames of motion, then still again. */
function clip(still: number, moving: number, after = still): Float32Array[] {
  const frames: Float32Array[] = [];
  const total = still + moving + after;
  for (let t = 0; t < total; t++) {
    const f = new Float32Array(FRAME_DIM);
    f[PRESENCE] = 1;
    const inside = t >= still && t < still + moving;
    f[0] = inside ? Math.sin(((t - still + 1) / (moving + 1)) * Math.PI) * 0.4 : 0;
    f[1] = inside ? 0.2 * ((t - still) % 2) : 0;
    frames.push(f);
  }
  return frames;
}

describe('motionEnergy', () => {
  it('is zero for a still mouth and for the first frame', () => {
    const e = motionEnergy(clip(10, 0));
    expect(e.every((v) => v === 0)).toBe(true);
  });

  it('ignores the jump into or out of a frame without a face', () => {
    const frames = clip(3, 0, 0);
    frames[1] = new Float32Array(FRAME_DIM);
    frames[0]![0] = 5;
    expect(motionEnergy(frames)).toEqual([0, 0, 0]);
  });
});

describe('trimByMotion', () => {
  it('keeps the moving part plus padding', () => {
    const result = trimByMotion(clip(12, 10), { pad: 2, minFrames: 5 });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.start).toBeGreaterThanOrEqual(9);
    expect(result.start).toBeLessThanOrEqual(11);
    expect(result.end).toBeGreaterThanOrEqual(23);
    expect(result.end).toBeLessThanOrEqual(25);
  });

  it('rejects a capture with no lip motion', () => {
    expect(trimByMotion(clip(20, 0), { pad: 2, minFrames: 5 })).toEqual({
      ok: false,
      reason: 'no-motion',
    });
  });

  it('rejects a capture that is too short', () => {
    expect(trimByMotion(clip(0, 3, 0), { pad: 0, minFrames: 5 })).toEqual({
      ok: false,
      reason: 'too-short',
    });
    expect(trimByMotion([], { pad: 2, minFrames: 5 })).toEqual({ ok: false, reason: 'too-short' });
  });

  it('never trims outside the capture', () => {
    const result = trimByMotion(clip(0, 12, 0), { pad: 4, minFrames: 5 });
    expect(result).toMatchObject({ ok: true, start: 0, end: 12 });
  });
});

describe('bestWindow', () => {
  it('returns the whole range when it fits', () => {
    expect(bestWindow([1, 2, 3], 0, 3, 5)).toEqual([0, 3]);
  });

  it('picks the window with the most motion', () => {
    const energy = [0, 0, 0, 0, 5, 5, 5, 0, 0, 0];
    expect(bestWindow(energy, 0, 10, 3)).toEqual([4, 7]);
  });
});
