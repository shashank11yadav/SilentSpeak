import { Camera, CameraOff, LoaderCircle, RotateCcw, Film } from 'lucide-react';
import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';
import type { TimedFrame } from '../ml/capture';
import { mouthOpening, trackingStatus, TRACKING_LABELS, type Tracking } from '../ml/tracking';
import {
  CAMERA_MESSAGES,
  cameraProblem,
  openCamera,
  stopStream,
  type CameraProblem,
} from '../lib/camera';
import { drawLips, pointsMapper } from '../lib/draw';
import { createTracker, type Tracker } from '../lib/landmarker';

type Phase = 'idle' | 'starting' | 'running' | 'paused' | 'error';
const MIN_FRAME_GAP_MS = 30;
const SPARK_POINTS = 120;

export interface CameraViewProps {
  base: string;
  /** While non-null, every processed frame is appended (the capture button is held). */
  recording: RefObject<TimedFrame[] | null>;
  openings: RefObject<number[]>;
  onTracking: (status: Tracking) => void;
  onUseReplay: () => void;
}

export function CameraView({
  base,
  recording,
  openings,
  onTracking,
  onUseReplay,
}: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const trackerRef = useRef<Tracker | null>(null);
  // Bumped by every start and stop, so a start still waiting on the camera
  // or the tracker can tell it was cancelled (tab hidden, Stop, unmount).
  const attempt = useRef(0);
  const wanted = useRef(false);
  const unmounted = useRef(false);
  const [phase, setPhase] = useState<Phase>('idle');
  const [problem, setProblem] = useState<CameraProblem | 'tracker' | null>(null);
  const [tracking, setTracking] = useState<Tracking>('idle');

  const report = useCallback(
    (status: Tracking) => {
      setTracking(status);
      onTracking(status);
    },
    [onTracking],
  );

  const stop = useCallback(
    (next: Phase) => {
      attempt.current += 1;
      wanted.current = false;
      stopStream(streamRef.current);
      streamRef.current = null;
      const video = videoRef.current;
      if (video) video.srcObject = null;
      setPhase(next);
      report('idle');
    },
    [report],
  );

  const start = useCallback(async () => {
    const mine = ++attempt.current;
    const cancelled = () => mine !== attempt.current;
    wanted.current = true;
    setProblem(null);
    setPhase('starting');
    let stream: MediaStream;
    try {
      stream = await openCamera();
    } catch (err) {
      if (cancelled()) return;
      wanted.current = false;
      setProblem(cameraProblem(err));
      setPhase('error');
      return;
    }
    // The tab was hidden or the camera stopped while the browser was asking.
    if (cancelled()) {
      stopStream(stream);
      return;
    }
    streamRef.current = stream;
    const video = videoRef.current;
    try {
      if (!video) throw new Error('video element missing');
      video.srcObject = stream;
      await video.play();
      if (!trackerRef.current) {
        const tracker = await createTracker(base);
        if (unmounted.current) {
          tracker.close();
          return;
        }
        trackerRef.current = tracker;
      }
      if (cancelled()) return; // stop() has already released the stream
      setPhase('running');
      report('no-face');
    } catch {
      if (cancelled()) return;
      wanted.current = false;
      stopStream(stream);
      streamRef.current = null;
      setProblem('tracker');
      setPhase('error');
    }
  }, [base, report]);

  // Release the camera whenever the tab is hidden or the page is left,
  // including while it is still starting.
  useEffect(() => {
    const onHide = () => {
      if (document.visibilityState === 'hidden' && wanted.current) stop('paused');
    };
    const onLeave = () => {
      if (wanted.current) stop('paused');
    };
    document.addEventListener('visibilitychange', onHide);
    window.addEventListener('pagehide', onLeave);
    return () => {
      document.removeEventListener('visibilitychange', onHide);
      window.removeEventListener('pagehide', onLeave);
      attempt.current += 1;
      wanted.current = false;
      stopStream(streamRef.current);
      streamRef.current = null;
    };
  }, [stop]);

  useEffect(() => {
    unmounted.current = false;
    return () => {
      unmounted.current = true;
      trackerRef.current?.close();
      trackerRef.current = null;
    };
  }, []);

  // Landmark loop: MediaPipe on the main thread, skipping frames it cannot keep up with.
  useEffect(() => {
    if (phase !== 'running') return;
    const video = videoRef.current;
    const canvas = overlayRef.current;
    const tracker = trackerRef.current;
    const ctx = canvas?.getContext('2d');
    if (!video || !canvas || !tracker || !ctx) return;
    let raf = 0;
    let lastRun = 0;
    let lastVideoTime = -1;
    let lastStatus: Tracking | null = null;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const now = performance.now();
      if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
      if (now - lastRun < MIN_FRAME_GAP_MS) return;
      lastRun = now;
      lastVideoTime = video.currentTime;
      const d = tracker.detect(video, now);
      const status = trackingStatus(d.faces, d.normalised);
      if (status !== lastStatus) {
        lastStatus = status;
        report(status);
      }
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (d.normalised && d.faces === 1) {
        const w = canvas.width;
        const h = canvas.height;
        drawLips(
          ctx,
          pointsMapper(d.normalised, (x, y) => [x * w, y * h]),
          Math.max(2, w / 260),
        );
      }
      const usable = status === 'tracking' ? d.frame : null;
      const series = openings.current;
      series.push(usable ? mouthOpening(usable.points) : 0);
      if (series.length > SPARK_POINTS) series.splice(0, series.length - SPARK_POINTS);
      recording.current?.push({ t: now, frame: usable });
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
    };
  }, [phase, report, recording, openings]);

  const running = phase === 'running';
  const message = problem === 'tracker' ? null : problem ? CAMERA_MESSAGES[problem] : null;

  return (
    <div className="relative aspect-[4/3] w-full overflow-hidden rounded-xl bg-stage text-white">
      <video
        ref={videoRef}
        className={`mirror absolute inset-0 h-full w-full object-cover ${running ? '' : 'invisible'}`}
        playsInline
        muted
        aria-label="Your camera, mirrored"
      />
      <canvas
        ref={overlayRef}
        className={`mirror pointer-events-none absolute inset-0 h-full w-full object-cover ${running ? '' : 'invisible'}`}
        aria-hidden="true"
      />

      {running ? (
        <>
          <div className="absolute top-3 left-3 flex items-center gap-2 rounded-full bg-black/55 px-3 py-1.5 text-xs font-medium backdrop-blur">
            <span
              className={`size-2 rounded-full ${tracking === 'tracking' ? 'bg-emerald-400' : 'bg-amber-400'}`}
              aria-hidden="true"
            />
            <span data-testid="tracking-status">{TRACKING_LABELS[tracking]}</span>
          </div>
          <button
            type="button"
            onClick={() => {
              stop('idle');
            }}
            className="absolute top-3 right-3 inline-flex items-center gap-1.5 rounded-full bg-black/55 px-3 py-1.5 text-xs font-medium backdrop-blur hover:bg-black/70"
          >
            <CameraOff className="size-3.5" aria-hidden="true" />
            Stop camera
          </button>
        </>
      ) : (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 p-6 text-center">
          {phase === 'starting' ? (
            <>
              <LoaderCircle className="size-8 animate-spin text-white/80" aria-hidden="true" />
              <p className="text-sm text-white/80" role="status">
                Starting the camera and face tracker…
              </p>
            </>
          ) : phase === 'error' ? (
            <div role="alert" className="max-w-sm space-y-3">
              <p className="text-lg font-semibold">
                {message ? message.title : 'The face tracker could not load'}
              </p>
              <p className="text-sm text-white/75">
                {message
                  ? message.body
                  : 'Your browser could not run MediaPipe. Try again, or use the replay clips instead.'}
              </p>
              <div className="flex flex-wrap justify-center gap-2 pt-1">
                <button
                  type="button"
                  onClick={() => void start()}
                  className="inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-sm font-semibold text-black hover:bg-white/90"
                >
                  <RotateCcw className="size-4" aria-hidden="true" />
                  Try again
                </button>
                <button
                  type="button"
                  onClick={onUseReplay}
                  className="inline-flex items-center gap-2 rounded-full border border-white/40 px-4 py-2 text-sm font-semibold hover:bg-white/10"
                >
                  <Film className="size-4" aria-hidden="true" />
                  Use replay clips
                </button>
              </div>
            </div>
          ) : (
            <>
              <div className="grid size-14 place-items-center rounded-full bg-white/10">
                <Camera className="size-7" aria-hidden="true" />
              </div>
              <div className="max-w-xs space-y-1">
                <p className="text-lg font-semibold">
                  {phase === 'paused' ? 'Camera paused' : 'Read your lips, privately'}
                </p>
                <p className="text-sm text-white/70">
                  {phase === 'paused'
                    ? 'The camera stops whenever this tab is hidden.'
                    : 'Video stays on this device. Only lip landmarks are used.'}
                </p>
              </div>
              <div className="flex flex-wrap justify-center gap-2">
                <button
                  type="button"
                  onClick={() => void start()}
                  className="inline-flex items-center gap-2 rounded-full bg-accent px-5 py-2.5 text-sm font-semibold text-accent-ink shadow-lg shadow-black/30 hover:bg-accent-strong"
                >
                  <Camera className="size-4" aria-hidden="true" />
                  {phase === 'paused' ? 'Resume camera' : 'Start camera'}
                </button>
                <button
                  type="button"
                  onClick={onUseReplay}
                  className="inline-flex items-center gap-2 rounded-full border border-white/30 px-4 py-2.5 text-sm font-semibold hover:bg-white/10"
                >
                  <Film className="size-4" aria-hidden="true" />
                  No camera? Replay
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
