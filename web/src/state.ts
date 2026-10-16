// App state for the three modes, as a pure reducer so the flows are testable.
import type { CategoryResult, PrototypeMatch, TaughtWord } from './ml/decide';
import type { Mode } from './ml/meta';
import {
  applyCommand,
  initialPlayer,
  suggestMapping,
  type Command,
  type Mapping,
  type PlayerState,
} from './lib/remote';
import { MAX_TAKES, MAX_WORDS } from './lib/word-store';

export interface Setup {
  labels: string[];
  takesPerWord: number;
  /** Word currently being recorded, and how many takes of it are done. */
  index: number;
  take: number;
}

export interface AppState {
  mode: Mode;
  categoryId: string | null;
  grid: { result: CategoryResult; truth: string | null; categoryId: string } | null;
  words: TaughtWord[];
  setup: Setup | null;
  teach: PrototypeMatch | null;
  mapping: Mapping;
  player: PlayerState;
  remote: { label: string | null; command: Command | null; sure: boolean } | null;
}

export type Action =
  | { type: 'mode'; mode: Mode }
  | { type: 'category'; id: string }
  | { type: 'words-loaded'; words: TaughtWord[] }
  | { type: 'grid-result'; result: CategoryResult; truth: string | null; categoryId: string }
  | { type: 'start-setup'; labels: string[]; takesPerWord: number }
  | { type: 'cancel-setup' }
  | { type: 'take'; embedding: Float32Array }
  | { type: 'teach-result'; match: PrototypeMatch }
  | { type: 'delete-word'; label: string }
  | { type: 'clear-words' }
  | { type: 'map'; command: Command; label: string | null }
  | { type: 'command'; command: Command }
  | { type: 'remote-result'; label: string | null; command: Command | null; sure: boolean };

export function initialState(mode: Mode, categoryId: string | null): AppState {
  return {
    mode,
    categoryId,
    grid: null,
    words: [],
    setup: null,
    teach: null,
    mapping: { playPause: null, next: null, previous: null },
    player: initialPlayer,
    remote: null,
  };
}

/** Keep mappings that still point at a taught word; suggest the rest. */
function reconcile(mapping: Mapping, words: readonly TaughtWord[]): Mapping {
  const labels = words.map((w) => w.label);
  const kept: Mapping = {
    playPause: mapping.playPause && labels.includes(mapping.playPause) ? mapping.playPause : null,
    next: mapping.next && labels.includes(mapping.next) ? mapping.next : null,
    previous: mapping.previous && labels.includes(mapping.previous) ? mapping.previous : null,
  };
  const free = labels.filter((l) => !Object.values(kept).includes(l));
  const suggested = suggestMapping(free);
  return {
    playPause: kept.playPause ?? suggested.playPause,
    next: kept.next ?? suggested.next,
    previous: kept.previous ?? suggested.previous,
  };
}

function withWords(state: AppState, words: TaughtWord[]): AppState {
  return { ...state, words, mapping: reconcile(state.mapping, words) };
}

function addTake(
  words: readonly TaughtWord[],
  label: string,
  embedding: Float32Array,
): TaughtWord[] {
  const existing = words.find((w) => w.label === label);
  if (existing) {
    return words.map((w) =>
      w.label === label ? { ...w, takes: [...w.takes, embedding].slice(-MAX_TAKES) } : w,
    );
  }
  if (words.length >= MAX_WORDS) return [...words];
  return [...words, { label, takes: [embedding] }];
}

export function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'mode':
      return { ...state, mode: action.mode };
    case 'category':
      return { ...state, categoryId: action.id, grid: null };
    case 'words-loaded':
      return withWords(state, action.words);
    case 'grid-result':
      return {
        ...state,
        grid: { result: action.result, truth: action.truth, categoryId: action.categoryId },
      };
    case 'start-setup': {
      const labels = action.labels.filter((l, i) => l && action.labels.indexOf(l) === i);
      if (labels.length === 0) return state;
      return {
        ...state,
        mode: 'teach',
        teach: null,
        setup: { labels, takesPerWord: action.takesPerWord, index: 0, take: 0 },
      };
    }
    case 'cancel-setup':
      return { ...state, setup: null };
    case 'take': {
      const setup = state.setup;
      const label = setup?.labels[setup.index];
      if (!setup || label === undefined) return state;
      const words = addTake(state.words, label, action.embedding);
      if (!words.some((w) => w.label === label)) return { ...state, setup: null };
      let { index, take } = setup;
      take += 1;
      if (take >= setup.takesPerWord) {
        index += 1;
        take = 0;
      }
      const done = index >= setup.labels.length;
      return { ...withWords(state, words), setup: done ? null : { ...setup, index, take } };
    }
    case 'teach-result':
      return { ...state, teach: action.match };
    case 'delete-word':
      return {
        ...withWords(
          state,
          state.words.filter((w) => w.label !== action.label),
        ),
        teach: null,
      };
    case 'clear-words':
      return { ...withWords(state, []), teach: null, setup: null, remote: null };
    case 'map':
      return { ...state, mapping: { ...state.mapping, [action.command]: action.label } };
    case 'command':
      return { ...state, player: applyCommand(state.player, action.command) };
    case 'remote-result':
      return {
        ...state,
        remote: { label: action.label, command: action.command, sure: action.sure },
        player:
          action.sure && action.command ? applyCommand(state.player, action.command) : state.player,
      };
  }
}
