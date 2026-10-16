// Camera access with messages a visitor can act on.

export type CameraProblem = 'denied' | 'not-found' | 'busy' | 'insecure' | 'unknown';

export const CAMERA_MESSAGES: Record<CameraProblem, { title: string; body: string }> = {
  denied: {
    title: 'Camera access is blocked',
    body: 'Allow the camera for this site in your browser settings and try again, or use the replay clips instead.',
  },
  'not-found': {
    title: 'No camera found',
    body: 'Connect a camera and try again, or use the replay clips instead.',
  },
  busy: {
    title: 'The camera is busy',
    body: 'Another app or tab may be using it. Close it and try again, or use the replay clips instead.',
  },
  insecure: {
    title: 'The camera needs a secure page',
    body: 'Open this demo over HTTPS (or localhost) to use the camera, or use the replay clips instead.',
  },
  unknown: {
    title: 'The camera could not start',
    body: 'Try again, or use the replay clips instead.',
  },
};

export function classifyCameraError(err: unknown): CameraProblem {
  const name = err instanceof DOMException || err instanceof Error ? err.name : '';
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
    case 'SecurityError':
      return 'denied';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
    case 'OverconstrainedError':
      return 'not-found';
    case 'NotReadableError':
    case 'TrackStartError':
    case 'AbortError':
      return 'busy';
    default:
      return 'unknown';
  }
}

export async function openCamera(): Promise<MediaStream> {
  if (!globalThis.isSecureContext || !(navigator.mediaDevices as MediaDevices | undefined)) {
    throw new DOMException('insecure context', 'InsecureContextError');
  }
  return navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
  });
}

export function cameraProblem(err: unknown): CameraProblem {
  if (err instanceof DOMException && err.name === 'InsecureContextError') return 'insecure';
  return classifyCameraError(err);
}

export function stopStream(stream: MediaStream | null) {
  stream?.getTracks().forEach((t) => {
    t.stop();
  });
}
