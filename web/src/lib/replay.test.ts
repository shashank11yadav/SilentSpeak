import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { prepareCapture } from '../ml/capture';
import { parseClipLandmarks, parseReplayIndex } from './replay';

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
