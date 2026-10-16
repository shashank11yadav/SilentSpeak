"""Where the pipeline reads data from and writes artefacts to."""

from __future__ import annotations

import os
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PIPELINE_DIR.parent
CONFIG_DIR = PIPELINE_DIR / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.toml"
REPORTS_DIR = PIPELINE_DIR / "reports"
RUNS_DIR = PIPELINE_DIR / "runs"
WEB_PUBLIC = REPO_ROOT / "web" / "public"
WEB_MODELS = WEB_PUBLIC / "models"
WEB_REPLAY = WEB_PUBLIC / "replay"
FIXTURES_DIR = REPO_ROOT / "shared" / "fixtures"


def default_data_dir() -> Path:
    """``$DATA_DIR`` if set, else ``../_datasets/grid`` next to the repo."""
    env = os.environ.get("DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT.parent / "_datasets" / "grid").resolve()


def raw_dir(data_dir: Path) -> Path:
    return data_dir / "raw"


def landmarks_dir(data_dir: Path) -> Path:
    return data_dir / "landmarks"


def prepared_path(data_dir: Path) -> Path:
    return data_dir / "prepared" / "segments.npz"
