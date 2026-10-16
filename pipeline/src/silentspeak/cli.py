"""Command line: ``uv run silentspeak <command>``."""

from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Annotated

import typer

from .config import Config
from .paths import (
    DEFAULT_CONFIG,
    FIXTURES_DIR,
    REPORTS_DIR,
    RUNS_DIR,
    WEB_MODELS,
    WEB_REPLAY,
    default_data_dir,
    landmarks_dir,
    prepared_path,
)

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
log = logging.getLogger("silentspeak")

DataDir = Annotated[
    Path | None,
    typer.Option(help="GRID data directory (default: $DATA_DIR, else ../_datasets/grid)."),
]
ConfigPath = Annotated[Path, typer.Option("--config", help="Run configuration (TOML).")]
Device = Annotated[str, typer.Option(help="auto, cpu, mps or cuda.")]


def _setup() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _data(data_dir: Path | None) -> Path:
    return (data_dir or default_data_dir()).resolve()


@app.command()
def extract(
    data_dir: DataDir = None,
    out_dir: Annotated[
        Path | None, typer.Option(help="Where sN.npz files go (default: <data-dir>/landmarks).")
    ] = None,
    speakers: Annotated[
        str | None,
        typer.Option(help="Comma-separated speakers, e.g. s1,s2 (default: all complete)."),
    ] = None,
    workers: Annotated[int, typer.Option(min=1, help="Parallel processes, one speaker each.")] = 1,
    limit_utterances: Annotated[
        int | None, typer.Option(min=1, help="Only the first N utterances per speaker (testing).")
    ] = None,
) -> None:
    """Run the MediaPipe face landmarker over every GRID frame, one .npz per speaker."""
    from .extract import extract_all

    _setup()
    data = _data(data_dir)
    chosen = [s.strip() for s in speakers.split(",")] if speakers else None
    stats = extract_all(
        data,
        out_dir or landmarks_dir(data),
        speakers=chosen,
        workers=workers,
        limit_utterances=limit_utterances,
    )
    typer.echo(json.dumps(stats, indent=2))


@app.command()
def prepare(
    data_dir: DataDir = None,
    config: ConfigPath = DEFAULT_CONFIG,
    landmarks: Annotated[Path | None, typer.Option(help="Extracted landmarks directory.")] = None,
) -> None:
    """Cut aligned word segments (plus padding) and cache their per-frame features."""
    from .dataset import prepare as run_prepare

    _setup()
    data = _data(data_dir)
    cfg = Config.load(config)
    segments = run_prepare(data, cfg, landmarks)
    out = prepared_path(data)
    segments.save(out)
    typer.echo(f"{len(segments)} segments from {len(set(segments.speakers))} speakers -> {out}")


@app.command()
def train(
    data_dir: DataDir = None,
    config: ConfigPath = DEFAULT_CONFIG,
    device: Device = "auto",
    run_dir: Annotated[Path, typer.Option(help="Checkpoint directory.")] = RUNS_DIR / "default",
    max_steps: Annotated[int | None, typer.Option(help="Stop after N steps (testing).")] = None,
) -> None:
    """Train the encoder; writes best.pt and train_log.json to --run-dir."""
    from .dataset import Segments
    from .train import train as run_train

    _setup()
    cfg = Config.load(config)
    segments = Segments.load(prepared_path(_data(data_dir)))
    best = run_train(segments, cfg, run_dir, device, max_steps=max_steps)
    typer.echo(f"best checkpoint: {best}")


@app.command()
def evaluate(
    data_dir: DataDir = None,
    config: ConfigPath = DEFAULT_CONFIG,
    checkpoint: Annotated[Path, typer.Option()] = RUNS_DIR / "default" / "best.pt",
    device: Device = "cpu",
    reports_dir: Annotated[Path, typer.Option()] = REPORTS_DIR,
) -> None:
    """Held-out-speaker metrics, few-shot accuracy, thresholds and the category gate."""
    from .dataset import Segments
    from .evaluate import evaluate as run_evaluate
    from .evaluate import metrics_markdown, write_reports

    _setup()
    cfg = Config.load(config)
    segments = Segments.load(prepared_path(_data(data_dir)))
    metrics = run_evaluate(segments, cfg, checkpoint, device)
    write_reports(metrics, reports_dir)
    typer.echo(metrics_markdown(metrics))


@app.command()
def export(
    stub: Annotated[bool, typer.Option(help="Random weights with the real I/O contract.")] = False,
    config: ConfigPath = DEFAULT_CONFIG,
    checkpoint: Annotated[Path, typer.Option()] = RUNS_DIR / "default" / "best.pt",
    metrics: Annotated[Path, typer.Option(help="metrics.json from evaluate.")] = REPORTS_DIR
    / "metrics.json",
    out_dir: Annotated[Path, typer.Option()] = WEB_MODELS,
) -> None:
    """Write model.onnx, model-meta.json and labels.json for the web app."""
    from .export import export_stub, export_trained
    from .train import load_checkpoint

    _setup()
    cfg = Config.load(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    if stub:
        path = export_stub(cfg, out_dir)
    else:
        model, words = load_checkpoint(checkpoint, cfg)
        if not metrics.exists():
            raise typer.BadParameter(f"{metrics} not found; run `silentspeak evaluate` first")
        path = export_trained(cfg, model, words, json.loads(metrics.read_text()), out_dir)
    typer.echo(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


@app.command("export-fixtures")
def export_fixtures(
    out: Annotated[Path, typer.Option()] = FIXTURES_DIR / "features.json",
) -> None:
    """Write the feature-parity fixture that the Python and TypeScript tests share."""
    from .export import write_fixture

    write_fixture(out)
    typer.echo(f"wrote {out}")


@app.command("replay-clips")
def replay_clips(
    data_dir: DataDir = None,
    config: ConfigPath = DEFAULT_CONFIG,
    out_dir: Annotated[Path, typer.Option()] = WEB_REPLAY,
    synthetic: Annotated[
        bool, typer.Option(help="Placeholder clips drawn from synthetic landmarks.")
    ] = False,
) -> None:
    """Build the six no-camera Replay clips (mouth-region frames plus landmarks)."""
    from .replay import build_grid_clips, build_synthetic_clips

    _setup()
    cfg = Config.load(config)
    if synthetic:
        index = build_synthetic_clips(cfg, out_dir)
    else:
        index = build_grid_clips(_data(data_dir), cfg, out_dir)
    typer.echo(f"wrote {len(index['clips'])} clips to {out_dir}")


@app.command()
def smoke(
    config: ConfigPath = DEFAULT_CONFIG.parent / "smoke.toml",
    keep: Annotated[Path | None, typer.Option(help="Keep outputs in this directory.")] = None,
) -> None:
    """Synthetic corpus -> prepare -> train -> evaluate -> export -> replay, in about a minute."""
    from .dataset import prepare as run_prepare
    from .evaluate import evaluate as run_evaluate
    from .evaluate import write_reports
    from .export import export_trained
    from .replay import build_synthetic_clips
    from .synthetic import write_corpus
    from .train import load_checkpoint
    from .train import train as run_train

    _setup()
    started = time.monotonic()
    cfg = Config.load(config)
    with tempfile.TemporaryDirectory(prefix="silentspeak-smoke-") as tmp:
        root = keep or Path(tmp)
        data = root / "grid"
        speakers = [f"s{i}" for i in range(1, 9)]
        write_corpus(data, speakers, utterances=20, seed=cfg.seed)
        segments = run_prepare(data, cfg)
        segments.save(prepared_path(data))
        best = run_train(segments, cfg, root / "run", device_name="cpu")
        metrics = run_evaluate(segments, cfg, best, "cpu")
        write_reports(metrics, root / "reports")
        model, words = load_checkpoint(best, cfg)
        onnx_path = export_trained(cfg, model, words, metrics, root / "models")
        index = build_synthetic_clips(cfg, root / "replay")
        elapsed = time.monotonic() - started
        typer.echo(
            f"smoke ok in {elapsed:.0f}s: {len(segments)} segments, "
            f"test 45-way top-1 {metrics['overall']['top1']:.2f}, "
            f"onnx {onnx_path.stat().st_size / 1e6:.2f} MB, {len(index['clips'])} replay clips"
        )


def main() -> None:
    app()
