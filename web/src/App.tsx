import { AlertTriangle, RotateCcw } from 'lucide-react';
import { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import { HowItWorks, SiteFooter } from './components/InfoSections';
import { GridWordsPanel } from './components/GridWordsPanel';
import { ModeTabs } from './components/ModeTabs';
import { RemotePanel } from './components/RemotePanel';
import { SiteHeader } from './components/SiteHeader';
import { Stage, type Source } from './components/Stage';
import { TeachPanel } from './components/TeachPanel';
import {
  FAILURE_MESSAGES,
  prepareCapture,
  type CaptureOptions,
  type TimedFrame,
} from './ml/capture';
import { categoryGuesses, nearestPrototype } from './ml/decide';
import { FPS } from './ml/features';
import {
  availableCategories,
  isPreview,
  openingMode,
  parseLabels,
  parseMeta,
  type Labels,
  type Mode,
  type ModelMeta,
} from './ml/meta';
import { ModelClient } from './lib/model-client';
import { commandFor } from './lib/remote';
import { loadReplayClips, type ReplayClip } from './lib/replay';
import { installTestHook } from './lib/test-hook';
import { WordStore } from './lib/word-store';
import { initialState, reducer, type AppState } from './state';

const BASE = import.meta.env.BASE_URL;

type ModelStatus =
  | { phase: 'loading' }
  | { phase: 'ready'; meta: ModelMeta; labels: Labels }
  | { phase: 'error'; message: string };

async function getJson(url: string): Promise<unknown> {
  const res = await fetch(url);
  if (!res.ok)
    throw new Error(`${url.split('/').pop() ?? url} could not be downloaded (HTTP ${res.status})`);
  return res.json();
}

export function App() {
  const [model, setModel] = useState<ModelStatus>({ phase: 'loading' });
  const [attempt, setAttempt] = useState(0);
  const clientRef = useRef<ModelClient | null>(null);
  const [state, dispatch] = useReducer(reducer, initialState('grid', null));
  const stateRef = useRef<AppState>(state);
  const [source, setSource] = useState<Source>('camera');
  const [clips, setClips] = useState<ReplayClip[] | null>(null);
  const [clipsError, setClipsError] = useState<string | null>(null);
  const [selectedClip, setSelectedClip] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ tone: 'info' | 'error'; text: string } | null>(null);
  const [persistent, setPersistent] = useState(true);
  const storeRef = useRef<WordStore | null>(null);
  const wordsReady = useRef(false);
  // The opening mode applies only until the visitor picks one themselves.
  const modeChosen = useRef(false);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  // Model files and the model worker. Retry re-runs this effect.
  useEffect(() => {
    let cancelled = false;
    const client = new ModelClient();
    clientRef.current = client;
    Promise.all([
      getJson(`${BASE}models/model-meta.json`),
      getJson(`${BASE}models/labels.json`),
      client.load(BASE),
    ]).then(
      ([metaJson, labelsJson]) => {
        if (cancelled) return;
        const meta = parseMeta(metaJson);
        const labels = parseLabels(labelsJson, meta);
        if (!modeChosen.current) dispatch({ type: 'mode', mode: openingMode(meta, labels) });
        const first = availableCategories(meta, labels)[0];
        if (first && stateRef.current.categoryId === null)
          dispatch({ type: 'category', id: first.id });
        setModel({ phase: 'ready', meta, labels });
      },
      (err: unknown) => {
        if (cancelled) return;
        setModel({ phase: 'error', message: err instanceof Error ? err.message : String(err) });
      },
    );
    return () => {
      cancelled = true;
      client.dispose();
      if (clientRef.current === client) clientRef.current = null;
    };
  }, [attempt]);

  useEffect(() => {
    loadReplayClips(BASE).then(
      (list) => {
        setClips(list);
        setSelectedClip((s) => s ?? list[0]?.id ?? null);
      },
      (err: unknown) => {
        setClipsError(err instanceof Error ? err.message : String(err));
      },
    );
  }, []);

  useEffect(() => {
    const store = new WordStore();
    storeRef.current = store;
    store.load().then(
      (words) => {
        wordsReady.current = true;
        dispatch({ type: 'words-loaded', words });
      },
      () => {
        wordsReady.current = true;
        setPersistent(false);
      },
    );
  }, []);

  useEffect(() => {
    if (!wordsReady.current || !persistent) return;
    storeRef.current?.saveAll(state.words).catch(() => {
      setPersistent(false);
    });
  }, [state.words, persistent]);

  const ready = model.phase === 'ready' ? model : null;

  const handleCapture = useCallback(
    (frames: TimedFrame[], opts: Partial<CaptureOptions>, clip: ReplayClip | null) => {
      const client = clientRef.current;
      if (!ready || !client) return;
      const prepared = prepareCapture(frames, {
        fps: FPS,
        minFrames: ready.meta.minFrames,
        maxFrames: ready.meta.maxFrames,
        pad: 2,
        ...opts,
      });
      if (!prepared.ok) {
        setMessage({ tone: 'error', text: FAILURE_MESSAGES[prepared.reason] });
        return;
      }
      setMessage(null);
      setBusy(true);
      client
        .infer(prepared.data, prepared.frames)
        .then((out) => {
          const s = stateRef.current;
          const { meta, labels } = ready;
          if (s.mode === 'grid') {
            const cats = availableCategories(meta, labels);
            let cat = cats.find((c) => c.id === s.categoryId) ?? cats[0];
            const clipCat = clip ? cats.find((c) => c.id === clip.category) : undefined;
            if (clipCat && clipCat.id !== cat?.id) {
              cat = clipCat;
              dispatch({ type: 'category', id: clipCat.id });
            }
            if (!cat) return;
            dispatch({
              type: 'grid-result',
              result: categoryGuesses(out.logits, meta.classes, cat.words, cat.threshold),
              truth: clip?.word ?? null,
              categoryId: cat.id,
            });
          } else if (s.setup) {
            dispatch({ type: 'take', embedding: out.embedding });
          } else if (s.mode === 'teach') {
            dispatch({
              type: 'teach-result',
              match: nearestPrototype(out.embedding, s.words, meta.thresholds),
            });
          } else {
            const match = nearestPrototype(out.embedding, s.words, meta.thresholds);
            const label = match.best?.label ?? null;
            dispatch({
              type: 'remote-result',
              label,
              command: label ? commandFor(s.mapping, label) : null,
              sure: match.sure,
            });
          }
        })
        .catch((err: unknown) => {
          setMessage({
            tone: 'error',
            text: `The model could not read that: ${err instanceof Error ? err.message : String(err)}`,
          });
        })
        .finally(() => {
          setBusy(false);
        });
    },
    [ready],
  );

  useEffect(() => {
    if (import.meta.env.MODE !== 'test') return;
    return installTestHook(handleCapture, clips);
  }, [handleCapture, clips]);

  const setMode = (mode: Mode) => {
    modeChosen.current = true;
    setMessage(null);
    dispatch({ type: 'mode', mode });
  };

  const preview = ready ? isPreview(ready.meta) : false;
  const categories = ready ? availableCategories(ready.meta, ready.labels) : [];
  const hiddenCategories =
    ready && !preview ? ready.labels.categories.filter((c) => !c.shipped) : [];
  // A clip from a hidden category (letters) would be scored against the wrong
  // words, so Replay offers only clips whose category ships.
  const offeredClips = (() => {
    if (!clips || !ready) return clips;
    const kept = clips.filter((c) => categories.some((k) => k.id === c.category));
    return kept.length > 0 ? kept : clips;
  })();
  const prompt = !ready
    ? 'Loading the lip-reading model…'
    : source === 'replay'
      ? 'The clip’s landmarks go through the same features and model as a live capture.'
      : state.setup
        ? `Mouth “${state.setup.labels[state.setup.index] ?? ''}” while holding.`
        : state.mode === 'grid'
          ? 'Hold, silently mouth one word from the list, release.'
          : state.mode === 'teach'
            ? state.words.length < 2
              ? 'Teach a few words first.'
              : 'Hold, mouth one of your words, release.'
            : 'Hold, mouth a control word, release.';
  const gridCredit = clips?.[0]?.source === 'grid';

  return (
    <>
      <a href="#main" className="skip-link">
        Skip to the demo
      </a>
      <SiteHeader />
      {preview && (
        <aside aria-label="Model status" className="border-b border-line bg-warn-soft">
          <p className="mx-auto max-w-6xl px-4 py-2 text-sm text-warn">
            <strong className="font-semibold">Untrained preview model.</strong> The app runs end to
            end, but the weights are random until training on GRID finishes, so GRID-word guesses
            are not meaningful yet.
          </p>
        </aside>
      )}
      <main id="main" tabIndex={-1} className="outline-none">
        <div className="mx-auto max-w-6xl px-4 pt-8 pb-4 sm:pt-10" id="top">
          <div className="max-w-3xl">
            <p className="text-sm font-semibold tracking-wide text-accent uppercase">
              Silent-word lip reading
            </p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight text-balance sm:text-4xl">
              Mouth a word without a sound. Watch it get read from your lips.
            </h1>
            <p className="mt-3 text-base leading-relaxed text-pretty text-ink-muted sm:text-lg">
              Face landmarks and a small lip-motion model run entirely in your browser. Try the
              51-word GRID vocabulary, teach it your own words in a minute, or drive a player with
              no hands and no voice.
            </p>
          </div>

          {model.phase === 'error' && (
            <div
              role="alert"
              className="mt-6 flex flex-col gap-3 rounded-xl border border-danger/40 bg-danger-soft p-4 text-danger sm:flex-row sm:items-center sm:justify-between"
            >
              <div className="flex gap-3">
                <AlertTriangle className="mt-0.5 size-5 shrink-0" aria-hidden="true" />
                <div>
                  <p className="font-semibold">The lip-reading model could not load.</p>
                  <p className="text-sm">{model.message}</p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => {
                  setModel({ phase: 'loading' });
                  setAttempt((a) => a + 1);
                }}
                className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg bg-danger px-4 py-2 text-sm font-semibold text-white hover:opacity-90"
              >
                <RotateCcw className="size-4" aria-hidden="true" /> Retry
              </button>
            </div>
          )}

          <div className="mt-6 grid items-start gap-5 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
            <Stage
              base={BASE}
              source={source}
              onSource={(s) => {
                setMessage(null);
                setSource(s);
              }}
              clips={offeredClips}
              clipsError={clipsError}
              selectedClip={selectedClip}
              onSelectClip={setSelectedClip}
              onCapture={handleCapture}
              busy={busy}
              disabled={!ready}
              message={message}
              prompt={prompt}
            />
            <section aria-label="Modes" className="card p-3 sm:p-4">
              <ModeTabs mode={state.mode} onMode={setMode} />
              <div
                id={`panel-${state.mode}`}
                role="tabpanel"
                aria-labelledby={`tab-${state.mode}`}
                className="px-1 pt-5 pb-2"
              >
                {!ready ? (
                  <p className="text-sm text-ink-muted" role="status">
                    {model.phase === 'error' ? 'Waiting for the model.' : 'Loading the model…'}
                  </p>
                ) : state.mode === 'grid' ? (
                  <GridWordsPanel
                    categories={categories}
                    hidden={hiddenCategories}
                    categoryId={state.categoryId}
                    onCategory={(id) => {
                      dispatch({ type: 'category', id });
                    }}
                    preview={preview}
                    result={state.grid}
                  />
                ) : state.mode === 'teach' ? (
                  <TeachPanel
                    words={state.words}
                    setup={state.setup}
                    match={state.teach}
                    persistent={persistent}
                    fewshot={
                      ready.meta.metrics && ready.meta.metrics.fewshotWords.length > 0
                        ? {
                            accuracy: ready.meta.metrics.fewshotAccuracy,
                            words: ready.meta.metrics.fewshotWords.length,
                          }
                        : null
                    }
                    onStartSetup={(labels, takesPerWord) => {
                      dispatch({ type: 'start-setup', labels, takesPerWord });
                    }}
                    onCancelSetup={() => {
                      dispatch({ type: 'cancel-setup' });
                    }}
                    onDelete={(label) => {
                      dispatch({ type: 'delete-word', label });
                    }}
                    onClear={() => {
                      dispatch({ type: 'clear-words' });
                    }}
                  />
                ) : (
                  <RemotePanel
                    words={state.words}
                    mapping={state.mapping}
                    player={state.player}
                    last={state.remote}
                    onMap={(command, label) => {
                      dispatch({ type: 'map', command, label });
                    }}
                    onCommand={(command) => {
                      dispatch({ type: 'command', command });
                    }}
                    onTeach={() => {
                      setMode('teach');
                    }}
                  />
                )}
              </div>
            </section>
          </div>
        </div>

        <div className="mx-auto mt-8 max-w-6xl px-4">
          <HowItWorks meta={ready?.meta ?? null} loaded={ready !== null} />
        </div>
      </main>
      <SiteFooter gridCredit={gridCredit} />
    </>
  );
}
