// Messages between the main thread and the model worker, and the checks the
// worker runs on everything it receives.
import { FEATURE_DIM } from '../ml/features';

export type WorkerRequest =
  | { type: 'load' }
  | { type: 'infer'; id: number; data: Float32Array; frames: number; featureDim: number };

export type WorkerResponse =
  | { type: 'loaded' }
  | { type: 'result'; id: number; logits: Float32Array; embedding: Float32Array }
  | { type: 'error'; id: number | null; message: string };

export interface ModelOutput {
  logits: Float32Array;
  embedding: Float32Array;
}

/**
 * Longest sequence the worker will run. The app never sends more than
 * model-meta.json's maxFrames (50); this only bounds what a bad message can
 * make the worker allocate.
 */
export const MAX_INFER_FRAMES = 1000;

export type ParsedRequest =
  { ok: true; request: WorkerRequest } | { ok: false; id: number | null; message: string };

const isObj = (x: unknown): x is Record<string, unknown> =>
  typeof x === 'object' && x !== null && !Array.isArray(x);

const isRequestId = (x: unknown): x is number => Number.isSafeInteger(x) && (x as number) > 0;

const isFrameCount = (x: unknown): x is number =>
  Number.isInteger(x) && (x as number) >= 1 && (x as number) <= MAX_INFER_FRAMES;

/**
 * Checks a message's shape before the worker acts on it. A bad inference
 * request comes back with its id when the id itself is valid, so the caller
 * can reject that one promise; anything else has no id to answer.
 */
export function parseWorkerRequest(msg: unknown): ParsedRequest {
  if (!isObj(msg)) return { ok: false, id: null, message: 'malformed message' };
  if (msg['type'] === 'load') return { ok: true, request: { type: 'load' } };
  if (msg['type'] !== 'infer') return { ok: false, id: null, message: 'unknown message type' };
  const id = msg['id'];
  if (!isRequestId(id)) return { ok: false, id: null, message: 'invalid request id' };
  const { data, frames, featureDim } = msg;
  const bad = (message: string): ParsedRequest => ({ ok: false, id, message });
  if (featureDim !== FEATURE_DIM) return bad('wrong feature size');
  if (!isFrameCount(frames)) return bad('frame count out of range');
  if (!(data instanceof Float32Array)) return bad('input is not a Float32Array');
  if (data.length !== frames * FEATURE_DIM) return bad('input has the wrong length');
  if (!data.every(Number.isFinite)) return bad('input is not finite');
  return { ok: true, request: { type: 'infer', id, data, frames, featureDim: FEATURE_DIM } };
}

// '/' or '/a/b/': a path on this origin, never '//host' or a full URL.
const SAME_ORIGIN_BASE = /^\/(?:[\w-]+\/)*$/;

/** Where the worker fetches the model: always a path on the page's own origin. */
export function modelUrl(base: string): string {
  if (!SAME_ORIGIN_BASE.test(base)) throw new Error(`refusing model base ${JSON.stringify(base)}`);
  return `${base}models/model.onnx`;
}
