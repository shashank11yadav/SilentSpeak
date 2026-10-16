// What the camera currently sees, for the status chip and to refuse captures
// that could only produce a confident wrong answer.
import {
  INNER_LOWER,
  INNER_UPPER,
  EYE_OUTER,
  N_STORED,
  STORED_INDICES,
  storedRow,
} from './landmarks';

export type Tracking = 'idle' | 'no-face' | 'many-faces' | 'partial' | 'tracking';

export const TRACKING_LABELS: Record<Tracking, string> = {
  idle: 'Camera off',
  'no-face': 'Looking for a face',
  'many-faces': 'More than one face: only one person, please',
  partial: 'Mouth partly out of frame',
  tracking: 'Tracking your lips',
};

/**
 * Classify one landmarker result. `normalised` holds the stored points as
 * MediaPipe returns them (0..1 across the image) for the first face.
 */
export function trackingStatus(faces: number, normalised: Float32Array | null): Tracking {
  if (faces === 0 || !normalised) return 'no-face';
  if (faces > 1) return 'many-faces';
  for (let n = 0; n < N_STORED; n++) {
    if (STORED_INDICES[n] === 152 || STORED_INDICES[n] === 1) continue; // chin and nose may crop
    const x = normalised[3 * n] ?? Number.NaN;
    const y = normalised[3 * n + 1] ?? Number.NaN;
    if (!(x >= 0 && x <= 1 && y >= 0 && y <= 1)) return 'partial';
  }
  return 'tracking';
}

/** Inner-lip gap divided by the outer-eye distance: the live "mouth opening" signal. */
export function mouthOpening(points: Float32Array): number {
  const p = (idx: number, c: number) => points[3 * storedRow(idx) + c] ?? 0;
  const gap = Math.hypot(
    p(INNER_UPPER, 0) - p(INNER_LOWER, 0),
    p(INNER_UPPER, 1) - p(INNER_LOWER, 1),
  );
  const eye = Math.hypot(
    p(EYE_OUTER[0], 0) - p(EYE_OUTER[1], 0),
    p(EYE_OUTER[0], 1) - p(EYE_OUTER[1], 1),
  );
  return eye > 0 ? gap / eye : 0;
}
