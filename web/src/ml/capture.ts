// From frames recorded while the capture button was held to the model input.
import { FPS, flatten, frameFeatures, resampleTimed, sequenceFeatures, PRESENCE } from './features';
import type { Frame } from './features';
import { bestWindow, motionEnergy, trimByMotion } from './trim';

export interface TimedFrame {
  /** Timestamp in milliseconds. */
  t: number;
  /** Landmarks, or null when no single face was found in this frame. */
  frame: Frame | null;
}

export interface CaptureOptions {
  fps: number;
  minFrames: number;
  maxFrames: number;
  pad: number;
  /** Auto-trim to the lip motion (live captures). Replay clips pass a fixed range. */
  trim?: boolean;
  range?: readonly [number, number];
}

export type CaptureFailure = 'too-short' | 'no-face' | 'no-motion';

export type PreparedCapture =
  { ok: true; data: Float32Array; frames: number } | { ok: false; reason: CaptureFailure };

// Captures where the face was found in fewer frames than this are rejected.
const MIN_PRESENT_FRACTION = 0.6;

export function prepareCapture(
  frames: readonly TimedFrame[],
  opts: CaptureOptions,
): PreparedCapture {
  if (frames.length < opts.minFrames) return { ok: false, reason: 'too-short' };
  const perFrame = frames.map((f) => frameFeatures(f.frame));
  let seq = resampleTimed(
    perFrame,
    frames.map((f) => f.t),
    opts.fps || FPS,
  );
  if (opts.range) seq = seq.slice(opts.range[0], opts.range[1]);
  const present = seq.filter((f) => f[PRESENCE] === 1).length;
  if (seq.length > 0 && present / seq.length < MIN_PRESENT_FRACTION) {
    return { ok: false, reason: 'no-face' };
  }
  if (opts.trim !== false) {
    const trimmed = trimByMotion(seq, { pad: opts.pad, minFrames: opts.minFrames });
    if (!trimmed.ok) return trimmed;
    seq = seq.slice(trimmed.start, trimmed.end);
  }
  if (seq.length < opts.minFrames) return { ok: false, reason: 'too-short' };
  const [a, b] = bestWindow(motionEnergy(seq), 0, seq.length, opts.maxFrames);
  seq = seq.slice(a, b);
  const features = sequenceFeatures(seq);
  return { ok: true, data: flatten(features), frames: features.length };
}

export const FAILURE_MESSAGES: Record<CaptureFailure, string> = {
  'too-short': 'That was too short to read. Hold the button for the whole word and say it again.',
  'no-face': 'We could not see your face for most of that. Face the camera and say it again.',
  'no-motion': 'We did not see your lips move. Mouth the word clearly and say it again.',
};
