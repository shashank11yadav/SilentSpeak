/// <reference lib="webworker" />
// Runs the ONNX lip-motion model off the main thread on ONNX Runtime's
// WebAssembly backend. The WebGPU build is deliberately not used: the model is
// small, its GRU has no WebGPU kernel and would run on the CPU anyway, and the
// WebGPU runtime would add a 28 MB wasm file to every visit.
import * as ort from 'onnxruntime-web/wasm';
import { modelUrl, parseWorkerRequest, type WorkerResponse } from '../lib/model-protocol';

declare const self: DedicatedWorkerGlobalScope;

// Fixed at build time, never taken from a message.
const MODEL_URL = modelUrl(import.meta.env.BASE_URL);

let session: ort.InferenceSession | null = null;

function post(msg: WorkerResponse, transfer: Transferable[] = []) {
  self.postMessage(msg, transfer);
}

async function load() {
  ort.env.wasm.numThreads = 1; // GitHub Pages cannot enable cross-origin isolation
  const res = await fetch(MODEL_URL, { mode: 'same-origin', credentials: 'same-origin' });
  if (!res.ok) throw new Error(`model download failed (HTTP ${res.status})`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
}

const errorText = (err: unknown) => (err instanceof Error ? err.message : String(err));

self.onmessage = (event: MessageEvent<unknown>) => {
  // Only the page that created this worker can post to it, and its messages
  // arrive with an empty origin (or the page's own, in some browsers). Check
  // anyway, so nothing else is ever acted on.
  if (event.origin !== '' && event.origin !== self.location.origin) return;
  const parsed = parseWorkerRequest(event.data);
  if (!parsed.ok) {
    // Without a valid id there is no request to answer; dropping it is safer
    // than failing a model load that is still in progress.
    if (parsed.id !== null) post({ type: 'error', id: parsed.id, message: parsed.message });
    return;
  }
  const msg = parsed.request;
  if (msg.type === 'load') {
    load().then(
      () => {
        post({ type: 'loaded' });
      },
      (err: unknown) => {
        post({ type: 'error', id: null, message: errorText(err) });
      },
    );
    return;
  }
  const { id, data, frames, featureDim } = msg;
  const current = session;
  if (!current) {
    post({ type: 'error', id, message: 'model not loaded' });
    return;
  }
  const input = new ort.Tensor('float32', data, [1, frames, featureDim]);
  current.run({ frames: input }).then(
    (out) => {
      const logits = Float32Array.from(out['logits']?.data as Float32Array);
      const embedding = Float32Array.from(out['embedding']?.data as Float32Array);
      post({ type: 'result', id, logits, embedding }, [logits.buffer, embedding.buffer]);
    },
    (err: unknown) => {
      post({ type: 'error', id, message: errorText(err) });
    },
  );
};
