import { describe, expect, it } from 'vitest';
import { prepareCapture, type TimedFrame } from './capture';
import { FEATURE_DIM, type Frame } from './features';
import { EYE_OUTER, N_BLENDSHAPES, N_STORED, OUTER_LOWER, storedRow } from './landmarks';

const OPTS = { fps: 25, minFrames: 5, maxFrames: 50, pad: 2 };

/** A plausible frontal face whose lower lip drops by `open` (image-height units). */
function face(open: number): Frame {
  const points = new Float32Array(N_STORED * 3);
  for (let n = 0; n < N_STORED; n++) {
    points[3 * n] = 0.5 + 0.002 * n;
    points[3 * n + 1] = 0.7;
  }
  for (const idx of OUTER_LOWER.slice(1, -1)) points[3 * storedRow(idx) + 1] = 0.72 + open;
  points[3 * storedRow(EYE_OUTER[0])] = 0.4;
  points[3 * storedRow(EYE_OUTER[0]) + 1] = 0.45;
  points[3 * storedRow(EYE_OUTER[1])] = 0.6;
  points[3 * storedRow(EYE_OUTER[1]) + 1] = 0.45;
  const matrix = new Float32Array(16);
  matrix[0] = matrix[5] = matrix[10] = matrix[15] = 1;
  return { points, blend: new Float32Array(N_BLENDSHAPES), matrix };
}

/** `still` frames, then a word lasting `moving` frames, then `still` again, at `fps`. */
function capture(still: number, moving: number, fps = 30, gone = false): TimedFrame[] {
  const out: TimedFrame[] = [];
  const total = 2 * still + moving;
  for (let i = 0; i < total; i++) {
    const inside = i >= still && i < still + moving;
    const open = inside ? 0.05 * Math.sin(((i - still + 1) / (moving + 1)) * Math.PI) : 0;
    out.push({ t: 1000 + (i * 1000) / fps, frame: gone ? null : face(open) });
  }
  return out;
}

describe('prepareCapture', () => {
  it('trims to the word and returns a model-ready buffer', () => {
    const result = prepareCapture(capture(15, 14), OPTS);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.frames).toBeGreaterThanOrEqual(OPTS.minFrames);
    expect(result.frames).toBeLessThan(20);
    expect(result.data.length).toBe(result.frames * FEATURE_DIM);
  });

  it('says so when no face was visible', () => {
    expect(prepareCapture(capture(10, 10, 30, true), OPTS)).toEqual({
      ok: false,
      reason: 'no-face',
    });
  });

  it('asks to say it again when the capture is too short', () => {
    expect(prepareCapture(capture(0, 3), OPTS)).toEqual({ ok: false, reason: 'too-short' });
    expect(prepareCapture([], OPTS)).toEqual({ ok: false, reason: 'too-short' });
  });

  it('says so when the lips did not move', () => {
    expect(prepareCapture(capture(20, 0), OPTS)).toEqual({ ok: false, reason: 'no-motion' });
  });

  it('caps very long captures at maxFrames', () => {
    const result = prepareCapture(capture(2, 90), OPTS);
    expect(result).toMatchObject({ ok: true, frames: OPTS.maxFrames });
  });

  it('uses an exact frame range when trimming is off (replay clips)', () => {
    const frames = capture(4, 10, 25);
    const result = prepareCapture(frames, { ...OPTS, trim: false, range: [2, 16] });
    expect(result).toMatchObject({ ok: true, frames: 14 });
  });
});
