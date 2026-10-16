import { Camera, Film, Mic, Play } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { CaptureOptions, TimedFrame } from '../ml/capture';
import type { Tracking } from '../ml/tracking';
import type { ReplayClip } from '../lib/replay';
import { CameraView } from './CameraView';
import { ReplayView, type LoadedClip } from './ReplayView';
import { Sparkline } from './Sparkline';

export type Source = 'camera' | 'replay';
const MAX_HOLD_MS = 4000;

interface Props {
  base: string;
  source: Source;
  onSource: (source: Source) => void;
  clips: ReplayClip[] | null;
  clipsError: string | null;
  selectedClip: string | null;
  onSelectClip: (id: string) => void;
  onCapture: (frames: TimedFrame[], opts: Partial<CaptureOptions>, clip: ReplayClip | null) => void;
  busy: boolean;
  disabled: boolean;
  message: { tone: 'info' | 'error'; text: string } | null;
  prompt: string;
}

function isTyping(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName);
}

export function Stage(props: Props) {
  const { source, onSource, onCapture, busy, disabled } = props;
  const recording = useRef<TimedFrame[] | null>(null);
  const openings = useRef<number[]>([]);
  const captureButton = useRef<HTMLButtonElement>(null);
  const holdTimer = useRef<number | undefined>(undefined);
  const [tracking, setTracking] = useState<Tracking>('idle');
  const [holding, setHolding] = useState(false);
  const [loadedClip, setLoadedClip] = useState<LoadedClip | null>(null);

  const cameraLive = source === 'camera' && tracking !== 'idle';
  const canHold = cameraLive && !busy && !disabled;

  const end = useCallback(() => {
    window.clearTimeout(holdTimer.current);
    const frames = recording.current;
    recording.current = null;
    setHolding(false);
    if (frames) onCapture(frames, { trim: true }, null);
  }, [onCapture]);

  const begin = useCallback(() => {
    if (!canHold || recording.current) return;
    recording.current = [];
    setHolding(true);
    // A word never takes this long; stop a stuck hold (e.g. a lost pointerup).
    holdTimer.current = window.setTimeout(end, MAX_HOLD_MS);
  }, [canHold, end]);

  // Space anywhere (outside text fields and other buttons) holds the capture.
  useEffect(() => {
    if (source !== 'camera') return;
    const down = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || e.repeat || isTyping(e.target)) return;
      const t = e.target;
      if (
        t instanceof HTMLElement &&
        t !== captureButton.current &&
        t.closest('button, a, [role="tab"], [role="radio"]')
      ) {
        return;
      }
      e.preventDefault();
      begin();
    };
    const up = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || !recording.current) return;
      e.preventDefault();
      end();
    };
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
    };
  }, [source, begin, end]);

  const onTracking = useCallback((status: Tracking) => {
    setTracking(status);
    if (status !== 'idle') captureButton.current?.focus({ preventScroll: true });
  }, []);

  const readClip = () => {
    if (!loadedClip) return;
    onCapture(
      loadedClip.landmarks.frames,
      { trim: false, range: loadedClip.clip.modelRange },
      loadedClip.clip,
    );
  };

  const tabClass = (active: boolean) =>
    `inline-flex flex-1 items-center justify-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium transition sm:flex-none ${
      active ? 'bg-surface text-ink shadow-sm' : 'text-ink-muted hover:text-ink'
    }`;

  return (
    <section aria-labelledby="stage-title" className="card p-3 sm:p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 id="stage-title" className="px-1 text-base font-semibold">
          {source === 'camera' ? 'Your camera' : 'Replay clips'}
        </h2>
        <div
          role="radiogroup"
          aria-label="Input"
          className="flex w-full gap-1 rounded-xl bg-surface-muted p-1 sm:w-auto"
        >
          <button
            type="button"
            role="radio"
            aria-checked={source === 'camera'}
            className={tabClass(source === 'camera')}
            onClick={() => {
              onSource('camera');
            }}
          >
            <Camera className="size-4" aria-hidden="true" /> Camera
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={source === 'replay'}
            className={tabClass(source === 'replay')}
            onClick={() => {
              onSource('replay');
            }}
          >
            <Film className="size-4" aria-hidden="true" /> Replay
          </button>
        </div>
      </div>

      {source === 'camera' ? (
        <CameraView
          base={props.base}
          recording={recording}
          openings={openings}
          onTracking={onTracking}
          onUseReplay={() => {
            onSource('replay');
          }}
        />
      ) : (
        <ReplayView
          clips={props.clips}
          error={props.clipsError}
          selected={props.selectedClip}
          onSelect={props.onSelectClip}
          onLoaded={setLoadedClip}
          openings={openings}
        />
      )}

      <div className="mt-3 rounded-xl border border-line px-3 pt-2 pb-1">
        <div className="flex items-center justify-between text-xs text-ink-muted">
          <span className="font-medium">Mouth opening</span>
          <span>last 4 s</span>
        </div>
        <Sparkline values={openings} />
      </div>

      <div className="mt-3">
        {source === 'camera' ? (
          <button
            ref={captureButton}
            type="button"
            disabled={!canHold && !holding}
            aria-pressed={holding}
            aria-describedby="capture-hint"
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId);
              begin();
            }}
            onPointerUp={() => {
              if (recording.current) end();
            }}
            onPointerCancel={() => {
              if (recording.current) end();
            }}
            onKeyDown={(e) => {
              if ((e.key === 'Enter' || e.key === ' ') && !e.repeat) {
                e.preventDefault();
                e.stopPropagation();
                begin();
              }
            }}
            onKeyUp={(e) => {
              if ((e.key === 'Enter' || e.key === ' ') && recording.current) {
                e.preventDefault();
                e.stopPropagation();
                end();
              }
            }}
            className={`flex w-full touch-none items-center justify-center gap-3 rounded-xl px-4 py-3.5 text-base font-semibold transition select-none disabled:cursor-not-allowed disabled:opacity-50 ${
              holding ? 'bg-danger text-white' : 'bg-accent text-accent-ink hover:bg-accent-strong'
            }`}
          >
            {holding ? (
              <>
                <span className="rec-dot size-2.5 rounded-full bg-white" aria-hidden="true" />
                Reading… release when you finish the word
              </>
            ) : busy ? (
              'Reading…'
            ) : (
              <>
                <Mic className="size-5" aria-hidden="true" />
                Hold to read
                <kbd className="hidden rounded-md border border-current/30 px-1.5 py-0.5 font-mono text-xs font-medium sm:inline">
                  Space
                </kbd>
              </>
            )}
          </button>
        ) : (
          <button
            ref={captureButton}
            type="button"
            onClick={readClip}
            disabled={!loadedClip || busy || disabled}
            className="flex w-full items-center justify-center gap-3 rounded-xl bg-accent px-4 py-3.5 text-base font-semibold text-accent-ink transition hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Play className="size-5" aria-hidden="true" />
            {busy ? 'Reading…' : 'Read this clip'}
          </button>
        )}
        <p id="capture-hint" className="mt-2 min-h-5 text-center text-sm text-ink-muted">
          {props.prompt}
        </p>
        <div aria-live="assertive" className="min-h-0">
          {props.message && (
            <p
              role={props.message.tone === 'error' ? 'alert' : 'status'}
              className={`mt-1 rounded-lg px-3 py-2 text-sm ${
                props.message.tone === 'error'
                  ? 'bg-warn-soft text-warn'
                  : 'bg-surface-muted text-ink-muted'
              }`}
            >
              {props.message.text}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
