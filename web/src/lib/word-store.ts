// Taught words live only in this browser's IndexedDB, as embeddings (128
// numbers per take), never as video or images.
import type { TaughtWord } from '../ml/decide';

const DB_NAME = 'silentspeak';
const STORE = 'words';
export const MAX_WORDS = 12;
export const MAX_TAKES = 5;

interface StoredWord {
  label: string;
  takes: Float32Array[];
  order: number;
}

function request<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => {
      resolve(req.result);
    };
    req.onerror = () => {
      reject(req.error ?? new Error('IndexedDB request failed'));
    };
  });
}

export class WordStore {
  private db: Promise<IDBDatabase> | null = null;

  constructor(private readonly idb: IDBFactory | undefined = globalThis.indexedDB) {}

  private open(): Promise<IDBDatabase> {
    const idb = this.idb;
    if (!idb) return Promise.reject(new Error('IndexedDB is not available'));
    this.db ??= new Promise((resolve, reject) => {
      const req = idb.open(DB_NAME, 1);
      req.onupgradeneeded = () => {
        req.result.createObjectStore(STORE, { keyPath: 'label' });
      };
      req.onsuccess = () => {
        resolve(req.result);
      };
      req.onerror = () => {
        reject(req.error ?? new Error('could not open IndexedDB'));
      };
    });
    return this.db;
  }

  private async tx(mode: IDBTransactionMode): Promise<IDBObjectStore> {
    const db = await this.open();
    return db.transaction(STORE, mode).objectStore(STORE);
  }

  async load(): Promise<TaughtWord[]> {
    const rows = (await request((await this.tx('readonly')).getAll())) as StoredWord[];
    return rows
      .sort((a, b) => a.order - b.order)
      .map((r) => ({ label: r.label, takes: r.takes.map((t) => Float32Array.from(t)) }));
  }

  /** Replace the stored list with `words`, keeping their order. */
  async saveAll(words: readonly TaughtWord[]): Promise<void> {
    const store = await this.tx('readwrite');
    await request(store.clear());
    await Promise.all(
      words.map((w, order) =>
        request(store.put({ label: w.label, takes: w.takes, order } satisfies StoredWord)),
      ),
    );
  }

  async clear(): Promise<void> {
    await request((await this.tx('readwrite')).clear());
  }
}

/** Clean up a typed label: trimmed, single spaces, at most 24 characters. */
export function normaliseLabel(raw: string): string {
  return raw.trim().replace(/\s+/g, ' ').slice(0, 24);
}
