import { Lock, ScanFace, Waves, Brain } from 'lucide-react';
import type { ModelMeta } from '../ml/meta';
import { GRID_URL, LIPNET_URL, MEDIAPIPE_URL, REPO_URL, RUN_LOCALLY_URL } from '../links';
import type { Backend } from '../lib/model-protocol';

const STEPS = [
  {
    Icon: ScanFace,
    title: 'Face landmarks',
    body: 'MediaPipe’s face landmarker finds 478 points on your face in each frame, on your device. The demo keeps 40 around the lips, plus the eye corners for scale.',
  },
  {
    Icon: Waves,
    title: 'Lip-motion features',
    body: 'Head turns are undone with the face’s pose matrix, sizes are normalised by eye distance, and left and right are folded together so a mirrored webcam reads the same. Frames are resampled to 25 per second.',
  },
  {
    Icon: Brain,
    title: 'A small model',
    body: 'Temporal convolutions and a bidirectional GRU turn the word into a 128-number signature. It names GRID words, and your own words are matched against your takes.',
  },
];

export function HowItWorks({ meta, backend }: { meta: ModelMeta | null; backend: Backend | null }) {
  return (
    <section id="how-it-works" aria-labelledby="how-title" className="scroll-mt-20">
      <h2 id="how-title" className="text-xl font-semibold tracking-tight">
        How it works
      </h2>
      <ol className="mt-4 grid gap-3 md:grid-cols-3">
        {STEPS.map(({ Icon, title, body }, i) => (
          <li key={title} className="card p-5">
            <div className="flex items-center gap-3">
              <span className="grid size-9 place-items-center rounded-lg bg-accent-soft text-accent">
                <Icon className="size-5" aria-hidden="true" />
              </span>
              <span className="text-xs font-semibold tracking-wide text-ink-muted uppercase">
                Step {i + 1}
              </span>
            </div>
            <h3 className="mt-3 font-semibold">{title}</h3>
            <p className="mt-1 text-sm leading-relaxed text-ink-muted">{body}</p>
          </li>
        ))}
      </ol>

      <div className="mt-3 grid gap-3 md:grid-cols-2">
        <div className="card p-5">
          <div className="flex items-center gap-2">
            <Lock className="size-4 text-accent" aria-hidden="true" />
            <h3 className="font-semibold">Private by design</h3>
          </div>
          <p className="mt-2 text-sm leading-relaxed text-ink-muted">
            No video, image or landmark leaves this page: there is no server, no analytics and no
            third-party request. The camera starts only when you ask and stops when you switch tabs.
            Taught words are stored in this browser as embeddings, and Clear all removes them.
          </p>
        </div>
        <div className="card p-5">
          <h3 className="font-semibold">Honest limits</h3>
          <p className="mt-2 text-sm leading-relaxed text-ink-muted">
            GRID is a fixed-grammar corpus of 51 words read in a studio, so “GRID words” only knows
            those words, and look-alike sounds (b, p, m) are hard for any lip reader. When the model
            is unsure it says so instead of guessing.{' '}
            {meta?.metrics
              ? `On held-out speakers it names the right word ${Math.round(meta.metrics.overallTop1 * 100)}% of the time across all 45 trained words.`
              : 'Held-out accuracy appears here after training.'}
          </p>
          <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-ink-muted">
            <dt>Model</dt>
            <dd className="text-ink">{meta?.metrics ? 'trained on GRID' : 'untrained preview'}</dd>
            <dt>Runs on</dt>
            <dd className="text-ink" data-testid="backend">
              {backend === 'webgpu' ? 'WebGPU' : backend === 'wasm' ? 'WebAssembly' : 'loading…'}
            </dd>
          </dl>
        </div>
      </div>
    </section>
  );
}

export function SiteFooter({ gridCredit }: { gridCredit: boolean }) {
  return (
    <footer className="mt-12 border-t border-line">
      <div className="mx-auto max-w-6xl space-y-2 px-4 py-8 text-sm text-ink-muted">
        <p>
          Training data and replay clips:{' '}
          <a href={GRID_URL} className="font-medium text-accent underline underline-offset-2">
            GRID corpus, Cooke et al., CC BY 4.0
          </a>
          {gridCredit
            ? ''
            : ' (the bundled replay clips are synthetic placeholders until the corpus is processed)'}
          . Face landmarks by{' '}
          <a href={MEDIAPIPE_URL} className="underline underline-offset-2">
            MediaPipe
          </a>{' '}
          (Apache 2.0); inference by ONNX Runtime Web (MIT).
        </p>
        <p>
          Related work:{' '}
          <a href={LIPNET_URL} className="underline underline-offset-2">
            LipNet (Assael et al., 2016)
          </a>{' '}
          reads whole GRID sentences from pixels; this demo reads single words from landmarks, small
          enough to run in a browser.
        </p>
        <p>
          <a href={REPO_URL} className="underline underline-offset-2">
            Source on GitHub
          </a>{' '}
          ·{' '}
          <a href={RUN_LOCALLY_URL} className="underline underline-offset-2">
            Run or retrain it locally
          </a>{' '}
          · MIT licence
        </p>
      </div>
    </footer>
  );
}
