import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  FEATURE_DIM,
  FRAME_DIM,
  PRESENCE,
  frameFeatures,
  resample,
  resampleTimed,
  sequenceFeatures,
  toHeightUnits,
  type Frame,
} from './features';
import { BLENDSHAPE_NAMES, STORED_INDICES } from './landmarks';

type Num = number | null;
interface FrameCase {
  name: string;
  points: Num[][];
  blend: Num[];
  matrix: Num[][];
  expected: number[];
}
interface Fixture {
  tolerance: number;
  frameDim: number;
  featureDim: number;
  storedIndices: number[];
  blendshapeNames: string[];
  frames: FrameCase[];
  sequence: {
    srcFps: number;
    dstFps: number;
    points: Num[][][];
    blend: Num[][];
    matrix: Num[][][];
    mirroredPoints: Num[][][];
    mirroredBlend: Num[][];
    mirroredMatrix: Num[][][];
    frameFeatures: number[][];
    resampled: number[][];
    expected: number[][];
  };
}

const fixture = JSON.parse(
  readFileSync(
    fileURLToPath(new URL('../../../shared/fixtures/features.json', import.meta.url)),
    'utf8',
  ),
) as Fixture;
const TOL = fixture.tolerance;

const f32 = (xs: Num[]) => Float32Array.from(xs, (x) => (x === null ? Number.NaN : x));
const frameOf = (points: Num[][], blend: Num[], matrix: Num[][]): Frame => ({
  points: f32(points.flat()),
  blend: f32(blend),
  matrix: f32(matrix.flat()),
});

function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>, tol = TOL) {
  expect(actual.length).toBe(expected.length);
  let worst = 0;
  for (let i = 0; i < actual.length; i++) {
    worst = Math.max(worst, Math.abs((actual[i] ?? 0) - (expected[i] ?? 0)));
  }
  expect(worst).toBeLessThanOrEqual(tol);
}

describe('constants', () => {
  it('match the Python pipeline', () => {
    expect(FRAME_DIM).toBe(fixture.frameDim);
    expect(FEATURE_DIM).toBe(fixture.featureDim);
    expect(PRESENCE).toBe(FRAME_DIM - 1);
    expect([...STORED_INDICES]).toEqual(fixture.storedIndices);
    expect([...BLENDSHAPE_NAMES]).toEqual(fixture.blendshapeNames);
  });
});

describe('frameFeatures (parity with Python)', () => {
  for (const c of fixture.frames) {
    it(`matches the fixture: ${c.name}`, () => {
      expectClose(frameFeatures(frameOf(c.points, c.blend, c.matrix)), c.expected);
    });
  }

  it('gives zeros for a frame without a face', () => {
    const missing = fixture.frames.find((c) => c.name === 'missing');
    expect(missing).toBeDefined();
    const f = frameFeatures(frameOf(missing!.points, missing!.blend, missing!.matrix));
    expect(f.every((v) => v === 0)).toBe(true);
  });

  it('treats a null frame as no face', () => {
    expect(frameFeatures(null).every((v) => v === 0)).toBe(true);
  });

  it('is identical for a mirrored face', () => {
    const open = fixture.frames.find((c) => c.name === 'open')!;
    const mirrored = fixture.frames.find((c) => c.name === 'open-mirrored')!;
    expectClose(
      frameFeatures(frameOf(mirrored.points, mirrored.blend, mirrored.matrix)),
      frameFeatures(frameOf(open.points, open.blend, open.matrix)),
    );
  });
});

describe('the full live path (parity with Python)', () => {
  const s = fixture.sequence;
  const run = (points: Num[][][], blend: Num[][], matrix: Num[][][]) =>
    points.map((p, i) => frameFeatures(frameOf(p, blend[i]!, matrix[i]!)));

  it('per-frame features', () => {
    run(s.points, s.blend, s.matrix).forEach((f, i) => {
      expectClose(f, s.frameFeatures[i]!);
    });
  });

  it('30 fps -> 25 fps resampling and deltas', () => {
    const feats = run(s.points, s.blend, s.matrix);
    const resampled = resample(feats, s.srcFps, s.dstFps);
    expect(resampled.length).toBe(s.resampled.length);
    resampled.forEach((f, i) => {
      expectClose(f, s.resampled[i]!);
    });
    sequenceFeatures(resampled).forEach((f, i) => {
      expectClose(f, s.expected[i]!);
    });
  });

  it('a mirrored webcam feed gives the same model input', () => {
    const feats = run(s.mirroredPoints, s.mirroredBlend, s.mirroredMatrix);
    sequenceFeatures(resample(feats, s.srcFps, s.dstFps)).forEach((f, i) => {
      expectClose(f, s.expected[i]!);
    });
  });

  it('resampling by timestamps equals resampling a uniform rate', () => {
    const feats = run(s.points, s.blend, s.matrix);
    const times = feats.map((_, i) => 1000 + (i * 1000) / s.srcFps);
    const timed = resampleTimed(feats, times, s.dstFps);
    expect(timed.length).toBe(s.resampled.length);
    timed.forEach((f, i) => {
      expectClose(f, s.resampled[i]!);
    });
  });
});

describe('resample', () => {
  const seq = (rows: number[][]) => rows.map((r) => Float32Array.from(r));

  it('is the identity at the same rate', () => {
    const x = seq([
      [1, 1],
      [2, 1],
      [3, 1],
    ]);
    expect(resample(x, 25, 25).map((r) => [...r])).toEqual([
      [1, 1],
      [2, 1],
      [3, 1],
    ]);
  });

  it('takes the nearer frame next to a frame without a face', () => {
    const x = seq([
      [2, 1],
      [0, 0],
      [4, 1],
    ]);
    expect(resample(x, 3, 4).map((r) => [...r])).toEqual([
      [2, 1],
      [0, 0],
      [4, 1],
    ]);
  });

  it('handles empty and single-frame input', () => {
    expect(resample([], 30, 25)).toEqual([]);
    expect(resample(seq([[5, 1]]), 30, 25).map((r) => [...r])).toEqual([[5, 1]]);
  });

  it('rejects a non-positive rate', () => {
    expect(() => resample(seq([[1, 1]]), 0, 25)).toThrow(/fps/);
  });

  it('copes with uneven webcam timestamps', () => {
    const x = seq([
      [0, 1],
      [10, 1],
      [20, 1],
    ]);
    // Frames at 0, 30 and 80 ms; outputs at 0, 40 and 80 ms.
    const out = resampleTimed(x, [0, 30, 80], 25);
    expect(out.map((r) => r[0])).toEqual([0, 12, 20]);
  });

  it('keeps every frame when timestamps repeat', () => {
    const x = seq([
      [1, 1],
      [2, 1],
    ]);
    expect(resampleTimed(x, [5, 5], 25).length).toBe(1);
  });
});

describe('toHeightUnits', () => {
  it('scales x and z by the aspect ratio', () => {
    const out = toHeightUnits(Float32Array.from([0.5, 0.25, -0.1]), 640, 480);
    expect(out[0]).toBeCloseTo(0.5 * (4 / 3), 6);
    expect(out[1]).toBeCloseTo(0.25, 6);
    expect(out[2]).toBeCloseTo(-0.1 * (4 / 3), 6);
  });
});
