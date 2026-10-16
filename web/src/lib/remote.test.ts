import { describe, expect, it } from 'vitest';
import { SCENES, applyCommand, commandFor, initialPlayer, suggestMapping } from './remote';

describe('silent remote', () => {
  it('play/pause toggles and next/previous wrap around', () => {
    let s = applyCommand(initialPlayer, 'playPause');
    expect(s.playing).toBe(true);
    s = applyCommand(s, 'previous');
    expect(s.scene).toBe(SCENES.length - 1);
    s = applyCommand(s, 'next');
    expect(s.scene).toBe(0);
    expect(s.pulse).toBe(3);
    expect(s.last).toBe('next');
  });

  it('suggests a mapping from taught words', () => {
    expect(suggestMapping(['Play', 'next', 'back', 'hello'])).toEqual({
      playPause: 'Play',
      next: 'next',
      previous: 'back',
    });
    expect(suggestMapping(['hello'])).toEqual({ playPause: null, next: null, previous: null });
  });

  it('maps a recognised word to its command', () => {
    const mapping = { playPause: 'play', next: 'next', previous: null };
    expect(commandFor(mapping, 'next')).toBe('next');
    expect(commandFor(mapping, 'hello')).toBeNull();
  });
});
