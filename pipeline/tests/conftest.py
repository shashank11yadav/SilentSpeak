from __future__ import annotations

from pathlib import Path

import pytest

from silentspeak.config import Config
from silentspeak.paths import CONFIG_DIR
from silentspeak.synthetic import write_corpus

SPEAKERS = ["s1", "s2", "s3", "s4", "s5", "s6"]


@pytest.fixture(scope="session")
def smoke_cfg() -> Config:
    return Config.load(CONFIG_DIR / "smoke.toml")


@pytest.fixture(scope="session")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data_dir = tmp_path_factory.mktemp("grid")
    write_corpus(data_dir, SPEAKERS, utterances=12, seed=3)
    return data_dir
