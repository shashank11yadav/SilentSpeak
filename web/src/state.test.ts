import { describe, expect, it } from 'vitest';
import { initialState, reducer, type AppState } from './state';

const e = (x: number) => Float32Array.from([x, 1 - x]);

function run(state: AppState, n: number): AppState {
  let s = state;
  for (let i = 0; i < n; i++) s = reducer(s, { type: 'take', embedding: e(i / 10) });
  return s;
}

describe('guided setup', () => {
  const started = reducer(initialState('grid', 'command'), {
    type: 'start-setup',
    labels: ['play', 'pause', 'next'],
    takesPerWord: 3,
  });

  it('switches to Teach and starts at the first word', () => {
    expect(started.mode).toBe('teach');
    expect(started.setup).toMatchObject({ index: 0, take: 0 });
  });

  it('records three takes per word, then moves on', () => {
    const s = run(started, 4);
    expect(s.setup).toMatchObject({ index: 1, take: 1 });
    expect(s.words.map((w) => [w.label, w.takes.length])).toEqual([
      ['play', 3],
      ['pause', 1],
    ]);
  });

  it('finishes after 3 words x 3 takes and suggests a remote mapping', () => {
    const s = run(started, 9);
    expect(s.setup).toBeNull();
    expect(s.words).toHaveLength(3);
    expect(s.mapping).toEqual({ playPause: 'play', next: 'next', previous: null });
  });

  it('ignores duplicate and empty labels', () => {
    const s = reducer(initialState('teach', null), {
      type: 'start-setup',
      labels: ['a', '', 'a', 'b'],
      takesPerWord: 3,
    });
    expect(s.setup?.labels).toEqual(['a', 'b']);
  });

  it('a take outside setup does nothing', () => {
    const s = initialState('teach', null);
    expect(reducer(s, { type: 'take', embedding: e(0) })).toBe(s);
  });
});

describe('words', () => {
  const taught = run(
    reducer(initialState('teach', null), {
      type: 'start-setup',
      labels: ['play', 'next'],
      takesPerWord: 1,
    }),
    2,
  );

  it('deleting a word clears its mapping', () => {
    const s = reducer(taught, { type: 'delete-word', label: 'next' });
    expect(s.words.map((w) => w.label)).toEqual(['play']);
    expect(s.mapping.next).toBeNull();
  });

  it('clear removes everything', () => {
    const s = reducer(taught, { type: 'clear-words' });
    expect(s.words).toEqual([]);
    expect(s.mapping).toEqual({ playPause: null, next: null, previous: null });
  });
});

describe('silent remote', () => {
  it('applies a recognised command only when sure', () => {
    const s0 = initialState('remote', null);
    const unsure = reducer(s0, {
      type: 'remote-result',
      label: 'next',
      command: 'next',
      sure: false,
    });
    expect(unsure.player.scene).toBe(0);
    const sure = reducer(s0, { type: 'remote-result', label: 'next', command: 'next', sure: true });
    expect(sure.player.scene).toBe(1);
    expect(sure.remote).toEqual({ label: 'next', command: 'next', sure: true });
  });
});
