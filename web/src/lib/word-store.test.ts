import { IDBFactory } from 'fake-indexeddb';
import { describe, expect, it } from 'vitest';
import { WordStore, normaliseLabel } from './word-store';

describe('WordStore', () => {
  it('round-trips taught words in order', async () => {
    const store = new WordStore(new IDBFactory());
    await store.saveAll([
      { label: 'play', takes: [Float32Array.from([1, 0]), Float32Array.from([0.9, 0.1])] },
      { label: 'next', takes: [Float32Array.from([0, 1])] },
    ]);
    const words = await store.load();
    expect(words.map((w) => w.label)).toEqual(['play', 'next']);
    expect([...words[0]!.takes[1]!]).toEqual([
      Float32Array.from([0.9])[0],
      Float32Array.from([0.1])[0],
    ]);
  });

  it('saveAll replaces and clear empties', async () => {
    const store = new WordStore(new IDBFactory());
    await store.saveAll([{ label: 'a', takes: [] }]);
    await store.saveAll([{ label: 'b', takes: [] }]);
    expect((await store.load()).map((w) => w.label)).toEqual(['b']);
    await store.clear();
    expect(await store.load()).toEqual([]);
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
