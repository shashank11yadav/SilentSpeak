import { describe, expect, it } from 'vitest';
import { FEATURE_DIM } from '../ml/features';
import { MAX_INFER_FRAMES, modelUrl, parseWorkerRequest } from './model-protocol';

const infer = (frames: number, extra: Record<string, unknown> = {}) => ({
  type: 'infer',
  id: 7,
  data: new Float32Array(frames * FEATURE_DIM),
  frames,
  featureDim: FEATURE_DIM,
  ...extra,
});

describe('parseWorkerRequest', () => {
  it('accepts a load request, which carries no URL', () => {
    expect(parseWorkerRequest({ type: 'load' })).toEqual({
      ok: true,
      request: { type: 'load' },
    });
  });

  it('ignores a URL sent along with a load request', () => {
    const parsed = parseWorkerRequest({ type: 'load', modelUrl: 'https://evil.example/m.onnx' });
    expect(parsed).toEqual({ ok: true, request: { type: 'load' } });
  });

  it('accepts a well-formed inference request', () => {
    const parsed = parseWorkerRequest(infer(12));
    expect(parsed.ok).toBe(true);
    if (parsed.ok && parsed.request.type === 'infer') {
      expect(parsed.request.id).toBe(7);
      expect(parsed.request.frames).toBe(12);
      expect(parsed.request.data).toBeInstanceOf(Float32Array);
    }
  });

  it.each([null, undefined, 3, 'load', [], {}, { type: 'run' }, { type: 'infer' }])(
    'rejects %j without a request id',
    (msg) => {
      expect(parseWorkerRequest(msg)).toMatchObject({ ok: false, id: null });
    },
  );

  it.each([
    ['a missing id', { id: undefined }],
    ['a fractional id', { id: 1.5 }],
    ['a negative id', { id: -1 }],
    ['a string id', { id: '7' }],
  ])('rejects %s and cannot name the request', (_, extra) => {
    expect(parseWorkerRequest(infer(5, extra))).toMatchObject({ ok: false, id: null });
  });

  it.each([
    ['data that is not a Float32Array', { data: Array.from({ length: 5 * FEATURE_DIM }, () => 0) }],
    ['a Float64Array', { data: new Float64Array(5 * FEATURE_DIM) }],
    ['a wrong feature size', { featureDim: FEATURE_DIM + 1 }],
    ['a length that does not match the shape', { data: new Float32Array(4 * FEATURE_DIM) }],
    ['zero frames', { frames: 0, data: new Float32Array(0) }],
    ['fractional frames', { frames: 2.5 }],
    [
      'too many frames',
      {
        frames: MAX_INFER_FRAMES + 1,
        data: new Float32Array((MAX_INFER_FRAMES + 1) * FEATURE_DIM),
      },
    ],
  ])('rejects %s and names the request so the caller hears back', (_, extra) => {
    expect(parseWorkerRequest(infer(5, extra))).toMatchObject({ ok: false, id: 7 });
  });

  it('rejects inputs that are not finite', () => {
    const msg = infer(5);
    msg.data[3] = Number.NaN;
    expect(parseWorkerRequest(msg)).toMatchObject({ ok: false, id: 7 });
    msg.data[3] = Number.POSITIVE_INFINITY;
    expect(parseWorkerRequest(msg)).toMatchObject({ ok: false, id: 7 });
  });
});

describe('modelUrl', () => {
  it('builds a same-origin path from the app base', () => {
    expect(modelUrl('/SilentSpeak/')).toBe('/SilentSpeak/models/model.onnx');
    expect(modelUrl('/')).toBe('/models/model.onnx');
  });

  it.each([
    'https://evil.example/',
    '//evil.example/',
    '/\\evil.example/',
    'SilentSpeak/',
    './',
    '/SilentSpeak/../',
    '/SilentSpeak',
    '',
  ])('refuses a base that could leave the origin: %j', (base) => {
    expect(() => modelUrl(base)).toThrow(/base/);
  });
});
