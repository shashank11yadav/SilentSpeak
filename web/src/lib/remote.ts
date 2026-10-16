// The "Silent remote" player: generated colour scenes, no media files.

export interface Scene {
  title: string;
  from: string;
  to: string;
}

export const SCENES: readonly Scene[] = [
  { title: 'Harbour at dawn', from: '#f6b38a', to: '#5a6fb0' },
  { title: 'Moss and fog', from: '#b9d7a8', to: '#2f5d50' },
  { title: 'Night market', from: '#f2c14e', to: '#7b2d5b' },
  { title: 'Glacier light', from: '#d6f0f5', to: '#3a7ca5' },
  { title: 'Desert road', from: '#f4d6a0', to: '#b4532a' },
  { title: 'Late train', from: '#a59ad6', to: '#1f2346' },
];

export type Command = 'playPause' | 'next' | 'previous';

const SUGGESTED: Record<Command, string[]> = {
  playPause: ['play', 'pause', 'stop'],
  next: ['next', 'skip'],
  previous: ['back', 'previous', 'prev'],
};

export const COMMANDS: readonly { id: Command; label: string }[] = [
  { id: 'playPause', label: 'Play / pause' },
  { id: 'next', label: 'Next' },
  { id: 'previous', label: 'Previous' },
];

export interface PlayerState {
  scene: number;
  playing: boolean;
  /** Increments on every command so the UI can replay its animation. */
  pulse: number;
  last: Command | null;
}

export const initialPlayer: PlayerState = { scene: 0, playing: false, pulse: 0, last: null };

export function applyCommand(state: PlayerState, command: Command): PlayerState {
  const n = SCENES.length;
  const base = { ...state, pulse: state.pulse + 1, last: command };
  switch (command) {
    case 'playPause':
      return { ...base, playing: !state.playing };
    case 'next':
      return { ...base, scene: (state.scene + 1) % n };
    case 'previous':
      return { ...base, scene: (state.scene - 1 + n) % n };
  }
}

export type Mapping = Record<Command, string | null>;

/** Default mapping: the first taught word that matches each command's suggestions. */
export function suggestMapping(labels: readonly string[]): Mapping {
  const used = new Set<string>();
  const pick = (suggested: string[]) => {
    const hit = labels.find((l) => suggested.includes(l.toLowerCase()) && !used.has(l)) ?? null;
    if (hit) used.add(hit);
    return hit;
  };
  return {
    playPause: pick(SUGGESTED.playPause),
    next: pick(SUGGESTED.next),
    previous: pick(SUGGESTED.previous),
  };
}

export function commandFor(mapping: Mapping, word: string): Command | null {
  for (const c of COMMANDS) if (mapping[c.id] === word) return c.id;
  return null;
}
