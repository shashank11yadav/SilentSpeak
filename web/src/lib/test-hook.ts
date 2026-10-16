// End-to-end tests drive the app with recorded landmark frames instead of a
// real face. Only installed when built with `--mode test`; App.tsx guards the
// call with import.meta.env.MODE so production bundles drop it.
import type { CaptureOptions, TimedFrame } from '../ml/capture';
import { loadClipLandmarks, type ReplayClip } from './replay';

type Capture = (
  frames: TimedFrame[],
  opts: Partial<CaptureOptions>,
  clip: ReplayClip | null,
) => void;

declare global {
  interface Window {
    __silentspeak?: {
      /**
       * Feed a replay clip's landmarks: as a live camera capture (auto-trimmed),
       * or as a replay clip. `limit` keeps only the first N frames.
       */
      captureClip: (id: string, as?: 'camera' | 'replay', limit?: number) => Promise<void>;
    };
  }
}

export function installTestHook(capture: Capture, clips: ReplayClip[] | null): () => void {
  window.__silentspeak = {
    async captureClip(id, as = 'camera', limit) {
      const clip = clips?.find((c) => c.id === id);
      if (!clip) throw new Error(`no clip ${id}`);
      const lm = await loadClipLandmarks(clip);
      const frames = lm.frames.slice(0, limit ?? lm.frames.length);
      if (as === 'camera') capture(frames, { trim: true }, null);
      else capture(frames, { trim: false, range: clip.modelRange }, clip);
    },
  };
  return () => {
    delete window.__silentspeak;
  };
}
