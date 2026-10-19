// Main-thread handle on the model worker.
import { FEATURE_DIM } from '../ml/features';
import type { ModelOutput, WorkerRequest, WorkerResponse } from './model-protocol';

export class ModelClient {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<
    number,
    { resolve: (o: ModelOutput) => void; reject: (e: Error) => void }
  >();
  private loading: { resolve: () => void; reject: (e: Error) => void } | null = null;

  constructor() {
    this.worker = new Worker(new URL('../workers/model.worker.ts', import.meta.url), {
      type: 'module',
      name: 'silentspeak-model',
    });
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      this.receive(event.data);
    };
    this.worker.onerror = (event) => {
      const err = new Error(event.message || 'model worker failed');
      this.loading?.reject(err);
      this.loading = null;
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
    };
  }

  private receive(msg: WorkerResponse) {
    if (msg.type === 'loaded') {
      this.loading?.resolve();
      this.loading = null;
    } else if (msg.type === 'result') {
      this.pending.get(msg.id)?.resolve({ logits: msg.logits, embedding: msg.embedding });
      this.pending.delete(msg.id);
    } else if (msg.id === null) {
      this.loading?.reject(new Error(msg.message));
      this.loading = null;
    } else {
      this.pending.get(msg.id)?.reject(new Error(msg.message));
      this.pending.delete(msg.id);
    }
  }

  private send(msg: WorkerRequest, transfer: Transferable[] = []) {
    this.worker.postMessage(msg, transfer);
  }

  /** The worker fetches the model from this site's own models/ folder. */
  load(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.loading = { resolve, reject };
      this.send({ type: 'load' });
    });
  }

  infer(data: Float32Array, frames: number): Promise<ModelOutput> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.send({ type: 'infer', id, data, frames, featureDim: FEATURE_DIM }, [data.buffer]);
    });
  }

  dispose() {
    this.worker.terminate();
  }
}
