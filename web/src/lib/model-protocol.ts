export type Backend = 'webgpu' | 'wasm';

export type WorkerRequest =
  | { type: 'load'; modelUrl: string; allowWebGpu: boolean }
  | { type: 'infer'; id: number; data: Float32Array; frames: number; featureDim: number };

export type WorkerResponse =
  | { type: 'loaded'; backend: Backend }
  | { type: 'result'; id: number; logits: Float32Array; embedding: Float32Array }
  | { type: 'error'; id: number | null; message: string };

export interface ModelOutput {
  logits: Float32Array;
  embedding: Float32Array;
}
