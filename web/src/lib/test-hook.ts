// End-to-end tests drive the app with recorded landmark frames instead of a
// real face. Only installed when built with `--mode test`; App.tsx guards the
// call with import.meta.env.MODE so production bundles drop it.
import type { CaptureOptions, TimedFrame } from '../ml/capture';
import { frameFeatures } from '../ml/features';
import { createImageLandmarker } from './landmarker';
import { loadClipLandmarks, type ReplayClip } from './replay';

/** What the browser makes of one still image, for the real-frame parity test. */
export interface ImageFeatures {
  width: number;
  height: number;
  faces: number;
  /** Stored points, MediaPipe-normalised, as [x, y, z] per point. */
  normalised: number[] | null;
  blend: number[] | null;
  /** The pose matrix after the app's column-major to row-major conversion. */
  matrix: number[] | null;
  /** frameFeatures() on the app's own Frame: the model's per-frame input. */
  features: number[];
  /** frameFeatures() with MediaPipe's flat matrix used as if it were row-major. */
  featuresUnconverted: number[];
}

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
      /** Landmark still images (IMAGE mode) and compute their features. */
      imageFeatures: (urls: string[], delegate?: 'CPU' | 'GPU') => Promise<ImageFeatures[]>;
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
    async imageFeatures(urls, delegate = 'CPU') {
      const landmarker = await createImageLandmarker(import.meta.env.BASE_URL, delegate);
      try {
        const out: ImageFeatures[] = [];
        for (const url of urls) {
          const res = await fetch(url);
          if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
          const bitmap = await createImageBitmap(await res.blob(), {
            colorSpaceConversion: 'none',
            premultiplyAlpha: 'none',
          });
          const { detection, rawMatrix } = landmarker.detect(bitmap);
          const frame = detection.frame;
          out.push({
            width: bitmap.width,
            height: bitmap.height,
            faces: detection.faces,
            normalised: detection.normalised ? Array.from(detection.normalised) : null,
            blend: frame ? Array.from(frame.blend) : null,
            matrix: frame ? Array.from(frame.matrix) : null,
            features: Array.from(frameFeatures(frame)),
            featuresUnconverted: Array.from(
              frameFeatures(
                frame && rawMatrix ? { ...frame, matrix: Float32Array.from(rawMatrix) } : null,
              ),
            ),
          });
          bitmap.close();
        }
        return out;
      } finally {
        landmarker.close();
      }
    },
  };
  return () => {
    delete window.__silentspeak;
  };
}
