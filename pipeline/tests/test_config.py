from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from silentspeak.config import Config
from silentspeak.paths import CONFIG_DIR
from silentspeak.vocab import ALL_WORDS, CATEGORIES


def test_vocabulary_is_51_words_in_six_categories() -> None:
    assert len(ALL_WORDS) == 51
    assert len(set(ALL_WORDS)) == 51
    assert list(CATEGORIES) == ["command", "colour", "preposition", "letter", "digit", "adverb"]
    assert "w" not in CATEGORIES["letter"]


@pytest.mark.parametrize("name", ["default.toml", "smoke.toml"])
def test_shipped_configs_load(name: str) -> None:
    cfg = Config.load(CONFIG_DIR / name)
    assert len(cfg.trained_words) == 45
    assert not set(cfg.trained_words) & set(cfg.data.fewshot_words)


def test_default_split_matches_the_spec() -> None:
    cfg = Config.load(CONFIG_DIR / "default.toml")
    assert cfg.data.test_speakers == ("s1", "s2", "s20", "s22", "s34")
    assert cfg.data.val_speakers == ("s3", "s4", "s23")
    assert cfg.data.fewshot_words == ("set", "white", "with", "k", "seven", "soon")


def test_rejects_overlapping_speakers(tmp_path: Path) -> None:
    text = (CONFIG_DIR / "default.toml").read_text()
    bad = text.replace('val_speakers = ["s3", "s4", "s23"]', 'val_speakers = ["s1"]')
    path = tmp_path / "bad.toml"
    path.write_text(bad)
    with pytest.raises(ValueError, match="both test and val"):
        Config.load(path)


def test_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text((CONFIG_DIR / "default.toml").read_text() + "\n[model2]\n")
    with pytest.raises(ValueError, match="unknown config sections"):
        Config.load(path)
    path.write_text(
        (CONFIG_DIR / "default.toml").read_text().replace("hidden = 192", "hidden = 192\nwidth = 3")
    )
    with pytest.raises(ValueError, match="unknown keys"):
        Config.load(path)


def test_replay_words_must_be_trained() -> None:
    cfg = Config.load(CONFIG_DIR / "default.toml")
    bad = dataclasses.replace(cfg, replay=dataclasses.replace(cfg.replay, words=("set",)))
    with pytest.raises(ValueError, match="trained"):
        bad.validate()
