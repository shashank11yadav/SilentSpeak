import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { IDBFactory } from 'fake-indexeddb';
import { describe, expect, it } from 'vitest';
import { EMBEDDING_DIM, MAX_TAKES, MAX_WORDS, WordStore, normaliseLabel } from './word-store';

const take = (v = 0.1) => new Float32Array(EMBEDDING_DIM).fill(v);

/** Write raw rows straight into the store, as another script or a corrupt profile could. */
async function seed(idb: IDBFactory, rows: unknown[]): Promise<void> {
  await new WordStore(idb).clear(); // creates the database and store
  const db = await new Promise<IDBDatabase>((resolve, reject) => {
    const req = idb.open('silentspeak', 1);
    req.onsuccess = () => {
      resolve(req.result);
    };
    req.onerror = () => {
      reject(new Error('open failed'));
    };
  });
  const tx = db.transaction('words', 'readwrite');
  for (const row of rows) tx.objectStore('words').put(row);
  await new Promise<void>((resolve) => {
    tx.oncomplete = () => {
      resolve();
    };
  });
  db.close();
}

describe('WordStore', () => {
  it('expects the embedding size the bundled model produces', () => {
    const meta = JSON.parse(
      readFileSync(
        fileURLToPath(new URL('../../public/models/model-meta.json', import.meta.url)),
        'utf8',
      ),
    ) as { embeddingDim: unknown };
    expect(meta.embeddingDim).toBe(EMBEDDING_DIM);
  });

  it('round-trips taught words in order', async () => {
    const store = new WordStore(new IDBFactory());
    await store.saveAll([
      { label: 'play', takes: [take(1), take(0.9)] },
      { label: 'next', takes: [take(0.5)] },
    ]);
    const words = await store.load();
    expect(words.map((w) => w.label)).toEqual(['play', 'next']);
    expect(words[0]!.takes[1]![5]).toBe(Float32Array.from([0.9])[0]);
  });

  it('saveAll replaces and clear empties', async () => {
    const store = new WordStore(new IDBFactory());
    await store.saveAll([{ label: 'a', takes: [take()] }]);
    await store.saveAll([{ label: 'b', takes: [take()] }]);
    expect((await store.load()).map((w) => w.label)).toEqual(['b']);
    await store.clear();
    expect(await store.load()).toEqual([]);
  });

  it('drops stored rows that are malformed instead of failing', async () => {
    const idb = new IDBFactory();
    await seed(idb, [
      { label: 'good', takes: [take()], order: 0 },
      { label: 'plain arrays', takes: [Array.from(take())], order: 1 },
      { label: 'no takes', takes: [], order: 2 },
      { label: 'takes not a list', takes: 'x', order: 3 },
      { label: 'short take', takes: [new Float32Array(3)], order: 4 },
      { label: 'not finite', takes: [take(Number.NaN)], order: 5 },
      { label: 'strings', takes: [Array.from({ length: EMBEDDING_DIM }, () => 'a')], order: 6 },
      { label: '   ', takes: [take()], order: 7 },
      { label: 'x'.repeat(25), takes: [take()], order: 8 },
      { label: 'no order', takes: [take()] },
      { label: 'wrong typed array', takes: [new Float64Array(EMBEDDING_DIM)], order: 9 },
      { label: 'object', takes: { length: 1, 0: take() }, order: 10 },
    ]);
    const words = await new WordStore(idb).load();
    expect(words.map((w) => w.label)).toEqual(['good', 'plain arrays', 'no order']);
    expect(words[1]!.takes[0]).toBeInstanceOf(Float32Array);
  });

  it('caps what it loads at the limits the app enforces', async () => {
    const idb = new IDBFactory();
    const rows = Array.from({ length: MAX_WORDS + 5 }, (_, i) => ({
      label: `w${String(i).padStart(2, '0')}`,
      takes: Array.from({ length: MAX_TAKES + 3 }, (_, k) => take(k + 1)),
      order: i,
    }));
    await seed(idb, rows);
    const words = await new WordStore(idb).load();
    expect(words).toHaveLength(MAX_WORDS);
    expect(words[0]!.label).toBe('w00');
    expect(words[0]!.takes).toHaveLength(MAX_TAKES);
    expect(words[0]!.takes[0]![0]).toBe(4); // the most recent takes are kept
  });

  it('ignores a row whose label is not its key', async () => {
    const idb = new IDBFactory();
    await seed(idb, [{ label: ['a'], takes: [take()], order: 0 }]);
    expect(await new WordStore(idb).load()).toEqual([]);
  });

  it('fails clearly without IndexedDB', async () => {
    await expect(new WordStore(undefined).load()).rejects.toThrow(/not available/);
  });
});

describe('normaliseLabel', () => {
  it('trims, collapses spaces and caps length', () => {
    expect(normaliseLabel('  turn   up ')).toBe('turn up');
    expect(normaliseLabel('x'.repeat(40))).toHaveLength(24);
  });
});
