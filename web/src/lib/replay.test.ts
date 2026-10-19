import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { prepareCapture } from '../ml/capture';
import { checkSprite, parseClipLandmarks, parseReplayIndex, type ReplayClip } from './replay';

const read = (name: string): unknown =>
  JSON.parse(
    readFileSync(fileURLToPath(new URL(`../../public/replay/${name}`, import.meta.url)), 'utf8'),
  );

describe('bundled replay clips', () => {
  const clips = parseReplayIndex(read('index.json'), '/SilentSpeak/');

  it('has six clips with safe same-origin URLs', () => {
    expect(clips).toHaveLength(6);
    for (const c of clips) {
      expect(c.spriteUrl).toMatch(/^\/SilentSpeak\/replay\/clip-\d\.jpg$/);
      expect(c.landmarksUrl).toMatch(/^\/SilentSpeak\/replay\/clip-\d\.json$/);
    }
  });

  it('each clip yields a model input over its model range', () => {
    for (const c of clips) {
      const lm = parseClipLandmarks(read(c.landmarksUrl.split('/').pop()!), c.fps);
      expect(lm.frames).toHaveLength(c.frames);
      const prepared = prepareCapture(lm.frames, {
        fps: 25,
        minFrames: 5,
        maxFrames: 50,
        pad: 2,
        trim: false,
        range: c.modelRange,
      });
      expect(prepared).toMatchObject({ ok: true, frames: c.modelRange[1] - c.modelRange[0] });
    }
  });
});

describe('parseReplayIndex', () => {
  it('refuses path tricks in file names', () => {
    const bad = {
      source: 'grid',
      clips: [
        {
          id: 'x',
          word: 'bin',
          category: 'command',
          speaker: 's1',
          fps: 25,
          modelRange: [0, 5],
          frames: 5,
          columns: 5,
          frameWidth: 10,
          frameHeight: 10,
          sprite: '../evil.jpg',
          landmarks: 'x.json',
        },
      ],
    };
    expect(() => parseReplayIndex(bad, '/')).toThrow(/sprite/);
  });

  it('marks frames without a face as null', () => {
    const row = (n: number, v: number | null) => Array.from({ length: n }, () => v);
    const lm = parseClipLandmarks(
      {
        frameSize: [360, 288],
        crop: [0, 0, 120, 80],
        points: [row(132, 0.5), row(132, null)],
        blend: [row(52, 0), row(52, null)],
        matrix: [row(16, 0), row(16, null)],
      },
      25,
    );
    expect(lm.frames[0]!.frame).not.toBeNull();
    expect(lm.frames[1]!.frame).toBeNull();
    expect(lm.frames[1]!.t).toBe(40);
  });
});

const clipJson = (extra: Record<string, unknown> = {}) => ({
  id: 'clip-1',
  word: 'bin',
  category: 'command',
  speaker: 's1',
  fps: 25,
  modelRange: [0, 5],
  frames: 5,
  columns: 5,
  frameWidth: 10,
  frameHeight: 10,
  sprite: 'clip-1.jpg',
  landmarks: 'clip-1.json',
  ...extra,
});

describe('parseReplayIndex bounds', () => {
  it('accepts a sane clip', () => {
    expect(parseReplayIndex({ source: 'grid', clips: [clipJson()] }, '/')).toHaveLength(1);
  });

  it.each([
    ['a zero frame rate', { fps: 0 }],
    ['a negative frame rate', { fps: -25 }],
    ['fractional frames', { frames: 5.5 }],
    ['zero columns', { columns: 0 }],
    ['more columns than frames', { columns: 6 }],
    ['a huge frame', { frameWidth: 100_000 }],
    ['a zero-height frame', { frameHeight: 0 }],
    ['too many frames', { frames: 100_000, modelRange: [0, 5] }],
    ['a fractional range', { modelRange: [0.5, 4] }],
    ['a range past the end', { modelRange: [0, 6] }],
  ])('rejects %s', (_, extra) => {
    expect(() => parseReplayIndex({ clips: [clipJson(extra)] }, '/')).toThrow(/invalid replay/);
  });

  it('rejects an index with an absurd number of clips', () => {
    const clips = Array.from({ length: 100 }, (_, i) => clipJson({ id: `c${i}` }));
    expect(() => parseReplayIndex({ clips }, '/')).toThrow(/invalid replay/);
  });
});

const row = (n: number, v: number | null) => Array.from({ length: n }, () => v);
const landmarks = (frames: number, extra: Record<string, unknown> = {}) => ({
  frameSize: [360, 288],
  crop: [0, 0, 120, 80],
  points: Array.from({ length: frames }, () => row(132, 0.5)),
  blend: Array.from({ length: frames }, () => row(52, 0)),
  matrix: Array.from({ length: frames }, () => row(16, 0)),
  ...extra,
});

describe('parseClipLandmarks bounds', () => {
  it('rejects a frame count that differs from the index', () => {
    expect(() => parseClipLandmarks(landmarks(4), 25, 5)).toThrow(/invalid replay/);
    expect(parseClipLandmarks(landmarks(5), 25, 5).frames).toHaveLength(5);
  });

  it.each([
    ['an empty crop', { crop: [0, 0, 0, 80] }],
    ['a negative frame size', { frameSize: [-360, 288] }],
    ['a short crop', { crop: [0, 0, 120] }],
  ])('rejects %s', (_, extra) => {
    expect(() => parseClipLandmarks(landmarks(2, extra), 25)).toThrow(/invalid replay/);
  });

  it('rejects coordinates that are not finite', () => {
    const lm = landmarks(2);
    (lm.points[1] as number[])[7] = Number.POSITIVE_INFINITY; // JSON.parse('1e999')
    expect(() => parseClipLandmarks(lm, 25)).toThrow(/invalid replay/);
  });
});

describe('checkSprite', () => {
  const clip = { frames: 19, columns: 8, frameWidth: 120, frameHeight: 80 } as ReplayClip;

  it('accepts a strip big enough for every frame', () => {
    expect(() => {
      checkSprite(clip, 960, 240);
    }).not.toThrow();
  });

  it('rejects a strip that is too small', () => {
    expect(() => {
      checkSprite(clip, 960, 160);
    }).toThrow(/sprite/);
    expect(() => {
      checkSprite(clip, 0, 0);
    }).toThrow(/sprite/);
  });
});
