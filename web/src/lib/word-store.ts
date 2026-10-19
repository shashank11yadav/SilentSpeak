// Taught words live only in this browser's IndexedDB, as embeddings (128
// numbers per take), never as video or images.
import type { TaughtWord } from '../ml/decide';

const DB_NAME = 'silentspeak';
const STORE = 'words';
export const MAX_WORDS = 12;
export const MAX_TAKES = 5;
/** Length of the model's embedding output (model-meta.json embeddingDim). */
export const EMBEDDING_DIM = 128;
// Rows read on load. Anything past this was not written by the app.
const MAX_ROWS = 4 * MAX_WORDS;

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

  /**
   * The stored words, validated: the database is outside the app's control
   * (an old version, another script on this origin, a corrupt profile), so
   * rows that do not look like what saveAll writes are dropped, and counts
   * are capped at the app's limits.
   */
  async load(): Promise<TaughtWord[]> {
    const rows = await request((await this.tx('readonly')).getAll(null, MAX_ROWS));
    const words: (TaughtWord & { order: number })[] = [];
    for (const row of rows) {
      const word = parseStoredWord(row);
      if (word && !words.some((w) => w.label === word.label)) words.push(word);
    }
    return words
      .sort((a, b) => a.order - b.order)
      .slice(0, MAX_WORDS)
      .map(({ label, takes }) => ({ label, takes }));
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

function parseTake(x: unknown): Float32Array | null {
  if (!(x instanceof Float32Array) && !Array.isArray(x)) return null;
  if (x.length !== EMBEDDING_DIM) return null;
  const values: unknown[] = Array.from(x);
  if (!values.every((v) => typeof v === 'number' && Number.isFinite(v))) return null;
  return Float32Array.from(values as number[]);
}

function parseStoredWord(row: unknown): (TaughtWord & { order: number }) | null {
  if (typeof row !== 'object' || row === null) return null;
  const { label, takes, order } = row as Partial<Record<keyof StoredWord, unknown>>;
  if (typeof label !== 'string' || !label || normaliseLabel(label) !== label) return null;
  if (!Array.isArray(takes)) return null;
  const valid = takes
    .slice(-MAX_TAKES)
    .map(parseTake)
    .filter((t): t is Float32Array => t !== null);
  if (valid.length === 0) return null;
  return {
    label,
    takes: valid,
    order: typeof order === 'number' && Number.isFinite(order) ? order : Number.MAX_VALUE,
  };
}

/** Clean up a typed label: trimmed, single spaces, at most 24 characters. */
export function normaliseLabel(raw: string): string {
  return raw.trim().replace(/\s+/g, ' ').slice(0, 24);
}
