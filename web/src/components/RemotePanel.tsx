import { Pause, Play, SkipBack, SkipForward } from 'lucide-react';
import type { TaughtWord } from '../ml/decide';
import { COMMANDS, SCENES, type Command, type Mapping, type PlayerState } from '../lib/remote';

interface Props {
  words: TaughtWord[];
  mapping: Mapping;
  player: PlayerState;
  last: { label: string | null; command: Command | null; sure: boolean } | null;
  onMap: (command: Command, label: string | null) => void;
  onCommand: (command: Command) => void;
  onTeach: () => void;
}

const COMMAND_LABEL: Record<Command, string> = {
  playPause: 'Play / pause',
  next: 'Next',
  previous: 'Previous',
};

export function RemotePanel({ words, mapping, player, last, onMap, onCommand, onTeach }: Props) {
  const scene = SCENES[player.scene] ?? { title: '', from: '#888', to: '#444' };
  const mapped = Object.values(mapping).filter(Boolean).length;
  return (
    <div className="space-y-5">
      <p className="text-sm text-ink-muted">
        Control a player without touching anything or making a sound: useful in a quiet library, a
        loud room, or with your hands busy.
      </p>

      <div
        key={player.pulse}
        className={`overflow-hidden rounded-xl border border-line ${player.pulse > 0 ? 'pulse' : ''}`}
        data-testid="player"
      >
        <div
          className="relative flex aspect-video items-end p-4 text-white"
          style={{ backgroundImage: `linear-gradient(135deg, ${scene.from}, ${scene.to})` }}
        >
          <div className="absolute top-3 right-3 rounded-full bg-black/35 px-2.5 py-1 text-xs font-medium backdrop-blur">
            {player.scene + 1} / {SCENES.length}
          </div>
          <div>
            <p className="text-xs font-medium tracking-wide text-white/80 uppercase">
              {player.playing ? 'Now playing' : 'Paused'}
            </p>
            <p className="text-xl font-semibold drop-shadow" data-testid="scene-title">
              {scene.title}
            </p>
          </div>
        </div>
        <div className="h-1 bg-surface-muted">
          <div
            className="scene-progress h-full bg-accent"
            data-playing={player.playing}
            key={player.scene}
          />
        </div>
        <div className="flex items-center justify-center gap-2 bg-surface p-2">
          <button
            type="button"
            aria-label="Previous scene"
            onClick={() => {
              onCommand('previous');
            }}
            className="grid size-9 place-items-center rounded-full hover:bg-surface-muted"
          >
            <SkipBack className="size-4" aria-hidden="true" />
          </button>
          <button
            type="button"
            aria-label={player.playing ? 'Pause' : 'Play'}
            onClick={() => {
              onCommand('playPause');
            }}
            className="grid size-11 place-items-center rounded-full bg-ink text-paper hover:opacity-90"
          >
            {player.playing ? (
              <Pause className="size-5" aria-hidden="true" />
            ) : (
              <Play className="size-5" aria-hidden="true" />
            )}
          </button>
          <button
            type="button"
            aria-label="Next scene"
            onClick={() => {
              onCommand('next');
            }}
            className="grid size-9 place-items-center rounded-full hover:bg-surface-muted"
          >
            <SkipForward className="size-4" aria-hidden="true" />
          </button>
        </div>
      </div>

      <div role="status" aria-live="polite" className="min-h-6 text-sm" data-testid="remote-result">
        {last &&
          (last.sure && last.command ? (
            <span>
              Heard <strong>{last.label}</strong> → {COMMAND_LABEL[last.command]}
            </span>
          ) : last.sure && last.label ? (
            <span className="text-ink-muted">
              Heard <strong>{last.label}</strong>, which is not mapped to a control.
            </span>
          ) : (
            <span className="text-ink-muted">Not sure what that was, so nothing happened.</span>
          ))}
      </div>

      {words.length < 2 ? (
        <div className="rounded-xl border border-dashed border-line p-4 text-sm">
          <p className="text-ink-muted">
            The remote uses words you teach it. Teach at least two (say “play” and “next”) first.
          </p>
          <button
            type="button"
            onClick={onTeach}
            className="mt-3 rounded-lg bg-ink px-3 py-2 text-sm font-semibold text-paper hover:opacity-90"
          >
            Teach words
          </button>
        </div>
      ) : (
        <fieldset className="space-y-2">
          <legend className="text-sm font-medium">
            Your words for each control <span className="text-ink-muted">({mapped} of 3 set)</span>
          </legend>
          {COMMANDS.map((c) => (
            <div key={c.id} className="grid grid-cols-[7rem_1fr] items-center gap-3 text-sm">
              <span className="text-ink-muted" aria-hidden="true">
                {c.label}
              </span>
              <select
                aria-label={c.label}
                value={mapping[c.id] ?? ''}
                onChange={(e) => {
                  onMap(c.id, e.target.value || null);
                }}
                className="rounded-lg border border-line bg-paper px-2.5 py-1.5"
              >
                <option value="">Not set</option>
                {words.map((w) => (
                  <option key={w.label} value={w.label}>
                    {w.label}
                  </option>
                ))}
              </select>
            </div>
          ))}
        </fieldset>
      )}
    </div>
  );
}
