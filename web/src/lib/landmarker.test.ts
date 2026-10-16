import { describe, expect, it } from 'vitest';
import { N_STORED } from '../ml/landmarks';
import { rowMajor, toDetection } from './landmarker';

const face = () => Array.from({ length: 478 }, (_, i) => ({ x: i / 478, y: 0.5, z: 0 }));
const identityCols = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 4, 5, 1];

describe('rowMajor', () => {
  it('moves the translation from the last column-major block to the last column', () => {
    const m = rowMajor(identityCols);
    expect([m[3], m[7], m[11]]).toEqual([3, 4, 5]);
    expect(m[12]).toBe(0);
  });
});

describe('toDetection', () => {
  const blend = [{ categories: [{ categoryName: 'jawOpen', score: 0.7 }] }];

  it('builds a frame for exactly one face', () => {
    const d = toDetection([face()], blend, [{ data: identityCols }], 640, 480);
    expect(d.faces).toBe(1);
    expect(d.frame?.points).toHaveLength(N_STORED * 3);
    expect(d.frame?.blend[25]).toBeCloseTo(0.7);
    expect(d.frame?.points[0]).toBeCloseTo(0 * (4 / 3));
  });

  it('refuses to build a frame for two faces', () => {
    const d = toDetection(
      [face(), face()],
      [...blend, ...blend],
      [{ data: identityCols }],
      640,
      480,
    );
    expect(d.faces).toBe(2);
    expect(d.frame).toBeNull();
    expect(d.normalised).not.toBeNull();
  });

  it('handles no face', () => {
    expect(toDetection([], [], [], 640, 480)).toEqual({ faces: 0, normalised: null, frame: null });
  });
});
