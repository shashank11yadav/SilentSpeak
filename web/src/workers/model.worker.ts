/// <reference lib="webworker" />
// Runs the ONNX lip-motion model off the main thread. WebGPU when the browser
// has it, WebAssembly otherwise; operators WebGPU lacks (the GRU) run on the
// CPU either way.
import * as ort from 'onnxruntime-web';
import type { WorkerRequest, WorkerResponse } from '../lib/model-protocol';

declare const self: DedicatedWorkerGlobalScope;

let session: ort.InferenceSession | null = null;

function post(msg: WorkerResponse, transfer: Transferable[] = []) {
  self.postMessage(msg, transfer);
}

async function create(bytes: Uint8Array, providers: string[]) {
  return ort.InferenceSession.create(bytes, {
    executionProviders: providers,
    graphOptimizationLevel: 'all',
  });
}

async function load(modelUrl: string, allowWebGpu: boolean) {
  ort.env.wasm.numThreads = 1; // GitHub Pages cannot enable cross-origin isolation
  const res = await fetch(modelUrl);
  if (!res.ok) throw new Error(`model download failed (HTTP ${res.status})`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  const hasGpu = allowWebGpu && 'gpu' in self.navigator;
  if (hasGpu) {
    try {
      session = await create(bytes, ['webgpu']);
      return 'webgpu' as const;
    } catch {
      // No usable adapter (or an unsupported driver): fall back to WebAssembly.
    }
  }
  session = await create(bytes, ['wasm']);
  return 'wasm' as const;
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  if (msg.type === 'load') {
    load(msg.modelUrl, msg.allowWebGpu).then(
      (backend) => {
        post({ type: 'loaded', backend });
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
