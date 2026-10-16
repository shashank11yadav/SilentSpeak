import { describe, expect, it } from 'vitest';
import { EYE_OUTER, INNER_LOWER, INNER_UPPER, N_STORED, storedRow } from './landmarks';
import { mouthOpening, trackingStatus } from './tracking';

const centred = () => {
  const p = new Float32Array(N_STORED * 3);
  for (let i = 0; i < N_STORED; i++) {
    p[3 * i] = 0.5;
    p[3 * i + 1] = 0.6;
  }
  return p;
};

describe('trackingStatus', () => {
  it('nothing in frame is not a face', () => {
    expect(trackingStatus(0, null)).toBe('no-face');
  });

  it('two faces are refused', () => {
    expect(trackingStatus(2, centred())).toBe('many-faces');
  });

  it('a mouth leaving the frame is partial', () => {
    const p = centred();
    p[3 * storedRow(61)] = 1.04;
    expect(trackingStatus(1, p)).toBe('partial');
  });

  it('a cropped chin is still fine', () => {
    const p = centred();
    p[3 * storedRow(152) + 1] = 1.2;
    expect(trackingStatus(1, p)).toBe('tracking');
  });
});

describe('mouthOpening', () => {
  it('is the inner-lip gap over the eye distance', () => {
    const p = centred();
    p[3 * storedRow(EYE_OUTER[0])] = 0.4;
    p[3 * storedRow(EYE_OUTER[1])] = 0.6;
    p[3 * storedRow(INNER_UPPER) + 1] = 0.6;
    p[3 * storedRow(INNER_LOWER) + 1] = 0.65;
    expect(mouthOpening(p)).toBeCloseTo(0.25, 5);
  });
});
