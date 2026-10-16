/// <reference lib="webworker" />
// Runs the ONNX lip-motion model off the main thread on ONNX Runtime's
// WebAssembly backend. The WebGPU build is deliberately not used: the model is
// small, its GRU has no WebGPU kernel and would run on the CPU anyway, and the
// WebGPU runtime would add a 28 MB wasm file to every visit.
import * as ort from 'onnxruntime-web/wasm';
import type { WorkerRequest, WorkerResponse } from '../lib/model-protocol';

declare const self: DedicatedWorkerGlobalScope;

let session: ort.InferenceSession | null = null;

function post(msg: WorkerResponse, transfer: Transferable[] = []) {
  self.postMessage(msg, transfer);
}

async function load(modelUrl: string) {
  ort.env.wasm.numThreads = 1; // GitHub Pages cannot enable cross-origin isolation
  const res = await fetch(modelUrl);
  if (!res.ok) throw new Error(`model download failed (HTTP ${res.status})`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  if (msg.type === 'load') {
    load(msg.modelUrl).then(
      () => {
        post({ type: 'loaded' });
      },
      (err: unknown) => {
        post({
          type: 'error',
          id: null,
          message: err instanceof Error ? err.message : String(err),
        });
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
      post({ type: 'error', id, message: err instanceof Error ? err.message : String(err) });
    },
  );
};
