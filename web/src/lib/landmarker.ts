// MediaPipe face landmarker, loaded on demand from this site's own files.
import { toHeightUnits, type Frame } from '../ml/features';
import { BLENDSHAPE_NAMES, N_BLENDSHAPES, STORED_INDICES } from '../ml/landmarks';

export interface Detection {
  faces: number;
  /** Stored points of the first face, MediaPipe-normalised (0..1), for drawing. */
  normalised: Float32Array | null;
  /** Model input for this frame; null unless exactly one face was found. */
  frame: Frame | null;
}

export interface Tracker {
  detect(video: HTMLVideoElement, timestampMs: number): Detection;
  readonly delegate: Delegate;
  close(): void;
}

interface Landmark {
  x: number;
  y: number;
  z: number;
}

/** MediaPipe's JS matrices arrive column-major; the features expect row-major. */
export function rowMajor(columnMajor: ArrayLike<number>): Float32Array {
  const out = new Float32Array(16);
  for (let r = 0; r < 4; r++)
    for (let c = 0; c < 4; c++) out[r * 4 + c] = columnMajor[c * 4 + r] ?? 0;
  return out;
}

const BLEND_INDEX = new Map<string, number>(BLENDSHAPE_NAMES.map((n, i) => [n, i]));

export function toDetection(
  faces: Landmark[][],
  blendshapes: { categories: { categoryName: string; score: number }[] }[],
  matrices: { data: number[] }[],
  width: number,
  height: number,
): Detection {
  const first = faces[0];
  if (!first) return { faces: 0, normalised: null, frame: null };
  const normalised = new Float32Array(STORED_INDICES.length * 3);
  STORED_INDICES.forEach((idx, n) => {
    const p = first[idx];
    normalised[3 * n] = p?.x ?? Number.NaN;
    normalised[3 * n + 1] = p?.y ?? Number.NaN;
    normalised[3 * n + 2] = p?.z ?? Number.NaN;
  });
  const cats = blendshapes[0]?.categories;
  const matrix = matrices[0]?.data;
  if (faces.length !== 1 || !cats || !matrix || matrix.length !== 16) {
    return { faces: faces.length, normalised, frame: null };
  }
  const blend = new Float32Array(N_BLENDSHAPES);
  for (const c of cats) {
    const i = BLEND_INDEX.get(c.categoryName);
    if (i !== undefined) blend[i] = c.score;
  }
  return {
    faces: 1,
    normalised,
    frame: { points: toHeightUnits(normalised, width, height), blend, matrix: rowMajor(matrix) },
  };
}

type Delegate = 'GPU' | 'CPU';

async function loadFaceLandmarker(base: string, mode: 'VIDEO' | 'IMAGE', delegate: Delegate) {
  const { FaceLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision');
  const origin = globalThis.location.origin;
  const fileset = await FilesetResolver.forVisionTasks(
    new URL(`${base}mediapipe/wasm`, origin).href,
  );
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: new URL(`${base}mediapipe/face_landmarker.task`, origin).href,
      delegate,
    },
    runningMode: mode,
    numFaces: 2,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  });
}

export async function createTracker(base: string): Promise<Tracker> {
  let delegate: Delegate = 'GPU';
  let task: Awaited<ReturnType<typeof loadFaceLandmarker>>;
  try {
    task = await loadFaceLandmarker(base, 'VIDEO', 'GPU');
  } catch {
    delegate = 'CPU';
    task = await loadFaceLandmarker(base, 'VIDEO', 'CPU');
  }
  let lastTs = -1;
  return {
    delegate,
    detect(video, timestampMs) {
      // MediaPipe requires strictly increasing timestamps.
      const ts = Math.max(Math.round(timestampMs), lastTs + 1);
      lastTs = ts;
      const r = task.detectForVideo(video, ts);
      return toDetection(
        r.faceLandmarks,
        r.faceBlendshapes,
        r.facialTransformationMatrixes,
        video.videoWidth,
        video.videoHeight,
      );
    },
    close() {
      task.close();
    },
  };
}

/**
 * The same landmarker for still images, one frame at a time. Used by the
 * real-frame parity test, which compares it with the Python landmarker.
 */
export async function createImageLandmarker(base: string, delegate: Delegate = 'CPU') {
  const task = await loadFaceLandmarker(base, 'IMAGE', delegate);
  return {
    detect(image: ImageBitmap): { detection: Detection; rawMatrix: number[] | null } {
      const r = task.detect(image);
      return {
        detection: toDetection(
          r.faceLandmarks,
          r.faceBlendshapes,
          r.facialTransformationMatrixes,
          image.width,
          image.height,
        ),
        rawMatrix: r.facialTransformationMatrixes[0]?.data.slice() ?? null,
      };
    },
    close() {
      task.close();
    },
  };
}
