// Browser-versus-Python parity on real GRID frames.
//
// shared/fixtures/grid-frames.json (written by `uv run silentspeak
// frame-fixture`) holds ten frames of held-out speaker s1 as lossless PNGs and
// what the Python MediaPipe landmarker and frame_features() make of them. Here
// the browser's MediaPipe landmarker reads the same PNGs, the app's own code
// turns the result into a Frame (including the pose matrix conversion), and
// web/src/ml/features.ts computes the features.
//
// The two MediaPipe builds differ slightly, so the comparison uses the
// tolerances stored in the fixture (see pipeline/src/silentspeak/realframes.py).
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';
import { ready, type ImageFeatures } from './helpers';

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'shared', 'fixtures');

interface FixtureFrame {
  file: string;
  frame: number;
  width: number;
  height: number;
  normalised: number[][];
  blend: number[];
  matrix: number[][];
  expected: number[];
}

interface Fixture {
  source: string;
  tolerance: { geometry: number; blendshapes: number; rotation: number };
  gpuTolerance: { geometry: number; blendshapes: number };
  frames: FixtureFrame[];
}

const fixture = JSON.parse(readFileSync(join(FIXTURES, 'grid-frames.json'), 'utf8')) as Fixture;

// Feature layout (features.ts): 62 lip-geometry values, 16 blendshape values,
// then the face-present flag.
const GEOMETRY = [0, 62] as const;
const BLEND = [62, 78] as const;
const PRESENCE = 78;

function maxDiff(a: readonly number[], b: readonly number[], [lo, hi]: readonly [number, number]) {
  let worst = 0;
  for (let i = lo; i < hi; i++) worst = Math.max(worst, Math.abs((a[i] ?? NaN) - (b[i] ?? NaN)));
  return worst;
}

function rotationDiff(rowMajor: readonly number[], python: number[][]) {
  let worst = 0;
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      worst = Math.max(worst, Math.abs((rowMajor[r * 4 + c] ?? NaN) - (python[r]?.[c] ?? NaN)));
  return worst;
}

async function measure(page: Page, delegate: 'CPU' | 'GPU') {
  await page.route('**/__fixtures/**', async (route) => {
    const file = new URL(route.request().url()).pathname.split('/__fixtures/')[1] ?? '';
    await route.fulfill({ body: readFileSync(join(FIXTURES, file)), contentType: 'image/png' });
  });
  await ready(page);
  await page.waitForFunction(() => window.__silentspeak !== undefined);
  const urls = fixture.frames.map((f) => `./__fixtures/${f.file}`);
  const got: ImageFeatures[] = await page.evaluate(
    ([list, d]) => window.__silentspeak?.imageFeatures(list, d) ?? Promise.resolve([]),
    [urls, delegate] as const,
  );
  expect(got).toHaveLength(fixture.frames.length);

  const rows = fixture.frames.map((want, i) => {
    const web = got[i];
    if (!web) throw new Error(`no browser result for ${want.file}`);
    expect(web.faces, want.file).toBe(1);
    expect([web.width, web.height]).toEqual([want.width, want.height]);
    expect(web.features[PRESENCE]).toBe(1);
    let landmarks = 0;
    want.normalised.forEach((p, n) => {
      for (let c = 0; c < 3; c++)
        landmarks = Math.max(
          landmarks,
          Math.abs((web.normalised?.[3 * n + c] ?? NaN) - (p[c] ?? NaN)),
        );
    });
    return {
      frame: want.frame,
      jawOpen: Number((want.blend[25] ?? 0).toFixed(3)),
      landmarks,
      rotation: rotationDiff(web.matrix ?? [], want.matrix),
      geometry: maxDiff(web.features, want.expected, GEOMETRY),
      blendshapes: maxDiff(web.features, want.expected, BLEND),
      unconvertedGeometry: maxDiff(web.featuresUnconverted, want.expected, GEOMETRY),
    };
  });
  const worst = (key: keyof (typeof rows)[number]) => Math.max(...rows.map((r) => r[key]));
  const summary = {
    delegate,
    frames: rows.length,
    landmarks: worst('landmarks'),
    rotation: worst('rotation'),
    geometry: worst('geometry'),
    blendshapes: worst('blendshapes'),
    unconvertedGeometry: worst('unconvertedGeometry'),
  };
  test.info().annotations.push({ type: 'parity', description: JSON.stringify(summary) });
  await test.info().attach(`parity-${delegate}.json`, {
    body: JSON.stringify({ summary, rows }, null, 2),
    contentType: 'application/json',
  });
  process.stdout.write(`real-frame parity: ${JSON.stringify(summary)}\n`);
  return summary;
}

test.describe('real-frame parity with the Python pipeline', () => {
  test.setTimeout(120_000);

  test('CPU landmarker: landmarks, pose matrix and features match Python', async ({ page }) => {
    const summary = await measure(page, 'CPU');
    const tol = fixture.tolerance;
    expect(summary.rotation).toBeLessThanOrEqual(tol.rotation);
    expect(summary.geometry).toBeLessThanOrEqual(tol.geometry);
    expect(summary.blendshapes).toBeLessThanOrEqual(tol.blendshapes);
    // Sharp enough to catch a matrix layout mistake: reading MediaPipe's
    // column-major data as row-major moves the features far outside the
    // tolerance, even on these nearly frontal GRID faces.
    expect(summary.unconvertedGeometry).toBeGreaterThan(5 * tol.geometry);
  });

  test('GPU landmarker (what the live camera tries first) stays close', async ({ page }) => {
    let summary: Awaited<ReturnType<typeof measure>>;
    try {
      summary = await measure(page, 'GPU');
    } catch (err) {
      // Browsers without WebGL fall back to the CPU delegate, checked above.
      test.skip(String(err).includes('WebGL'), 'no GPU delegate in this browser');
      throw err;
    }
    const tol = fixture.gpuTolerance;
    expect(summary.geometry).toBeLessThanOrEqual(tol.geometry);
    expect(summary.blendshapes).toBeLessThanOrEqual(tol.blendshapes);
  });
});
