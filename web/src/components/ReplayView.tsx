import { LoaderCircle } from 'lucide-react';
import { useEffect, useRef, useState, type RefObject } from 'react';
import { CATEGORY_LABELS } from '../ml/meta';
import { mouthOpening } from '../ml/tracking';
import { drawLips, pointsMapper } from '../lib/draw';
import { checkSprite, loadClipLandmarks, type ClipLandmarks, type ReplayClip } from '../lib/replay';

const SCALE = 4; // canvas pixels per sprite pixel

export interface LoadedClip {
  clip: ReplayClip;
  landmarks: ClipLandmarks;
}

interface Props {
  clips: ReplayClip[] | null;
  error: string | null;
  selected: string | null;
  onSelect: (id: string) => void;
  onLoaded: (loaded: LoadedClip | null) => void;
  openings: RefObject<number[]>;
}

export function ReplayView({ clips, error, selected, onSelect, onLoaded, openings }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loaded, setLoaded] = useState<{
    id: string;
    lm: ClipLandmarks;
    img: HTMLImageElement;
  } | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const clip = clips?.find((c) => c.id === selected) ?? null;

  useEffect(() => {
    if (!clip) return;
    let cancelled = false;
    const img = new Image();
    img.decoding = 'async';
    img.src = clip.spriteUrl;
    Promise.all([loadClipLandmarks(clip), img.decode()])
      .then(([lm]) => {
        checkSprite(clip, img.naturalWidth, img.naturalHeight);
        return lm;
      })
      .then(
        (lm) => {
          if (cancelled) return;
          setLoadError(null);
          setLoaded({ id: clip.id, lm, img });
          onLoaded({ clip, landmarks: lm });
        },
        (err: unknown) => {
          if (cancelled) return;
          setLoadError(err instanceof Error ? err.message : String(err));
          onLoaded(null);
        },
      );
    return () => {
      cancelled = true;
    };
  }, [clip, onLoaded]);

  // Loop the frame strip at the clip's frame rate with the lip overlay on top.
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!clip || !loaded || loaded.id !== clip.id || !canvas || !ctx) return;
    const { lm, img } = loaded;
    const w = clip.frameWidth * SCALE;
    const h = clip.frameHeight * SCALE;
    canvas.width = w;
    canvas.height = h;
    const [, frameH] = lm.frameSize;
    const [cx, cy, cw, ch] = lm.crop;
    const toCanvas = (x: number, y: number): [number, number] => [
      ((x * frameH - cx) / cw) * w,
      ((y * frameH - cy) / ch) * h,
    ];
    let raf = 0;
    let frame = -1;
    const started = performance.now();
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      const i = reduced
        ? Math.floor((clip.modelRange[0] + clip.modelRange[1]) / 2)
        : Math.floor(((now - started) / 1000) * clip.fps) % clip.frames;
      if (i === frame) return;
      frame = i;
      const sx = (i % clip.columns) * clip.frameWidth;
      const sy = Math.floor(i / clip.columns) * clip.frameHeight;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, sx, sy, clip.frameWidth, clip.frameHeight, 0, 0, w, h);
      const f = lm.frames[i]?.frame;
      if (f) {
        drawLips(ctx, pointsMapper(f.points, toCanvas), 3);
        const series = openings.current;
        series.push(mouthOpening(f.points));
        if (series.length > 120) series.splice(0, series.length - 120);
      }
      const inWord = i >= clip.modelRange[0] && i < clip.modelRange[1];
      ctx.fillStyle = inWord ? 'rgba(242, 122, 160, 0.95)' : 'rgba(255, 255, 255, 0.35)';
      ctx.fillRect(0, h - 6, ((i + 1) / clip.frames) * w, 6);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
    };
  }, [clip, loaded, openings]);

  const ready = loaded !== null && clip !== null && loaded.id === clip.id;

  return (
    <div>
      <div className="relative aspect-[3/2] w-full overflow-hidden rounded-xl bg-stage">
        <canvas
          ref={canvasRef}
          className={`absolute inset-0 h-full w-full ${ready ? '' : 'invisible'}`}
          role="img"
          aria-label={
            clip
              ? `Replay clip of the word “${clip.word}” with the lip contour drawn on top`
              : 'Replay clip'
          }
        />
        {!ready && (
          <div className="absolute inset-0 grid place-items-center p-6 text-center text-sm text-white/80">
            {error || loadError ? (
              <p role="alert">Replay clips could not load: {error ?? loadError}</p>
            ) : clips && !clip ? (
              <p>Pick a clip below.</p>
            ) : (
              <LoaderCircle className="size-7 animate-spin" aria-label="Loading clip" />
            )}
          </div>
        )}
        {clip && ready && (
          <div className="absolute top-3 left-3 rounded-full bg-black/55 px-3 py-1.5 text-xs font-medium text-white backdrop-blur">
            {clip.source === 'grid' ? `GRID speaker ${clip.speaker}` : 'Synthetic placeholder'}
          </div>
        )}
      </div>
      {clips && (
        <div
          role="radiogroup"
          aria-label="Replay clip"
          className={`mt-3 grid grid-cols-3 gap-2 ${clips.length === 5 ? 'sm:grid-cols-5' : 'sm:grid-cols-6'}`}
        >
          {clips.map((c) => {
            const active = c.id === selected;
            return (
              <button
                key={c.id}
                type="button"
                role="radio"
                aria-checked={active}
                onClick={() => {
                  onSelect(c.id);
                }}
                className={`rounded-lg border px-2 py-1.5 text-left text-xs transition ${
                  active
                    ? 'border-accent bg-accent-soft text-ink'
                    : 'border-line bg-surface hover:border-ink-muted/50'
                }`}
              >
                <span className="block font-semibold">{c.word}</span>
                <span className="block text-ink-muted">
                  {CATEGORY_LABELS[c.category] ?? c.category}
                </span>
              </button>
            );
          })}
        </div>
      )}
      {clips?.[0]?.credit && (
        <p className="mt-2 text-xs text-ink-muted">
          Clips: {clips[0].credit}. Speakers the model never trained on.
        </p>
      )}
    </div>
  );
}
