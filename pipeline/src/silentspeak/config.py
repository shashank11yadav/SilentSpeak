"""Typed run configuration loaded from a TOML file (``configs/default.toml``)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Self

from .vocab import ALL_WORDS, CATEGORIES


@dataclass(frozen=True)
class DataConfig:
    test_speakers: tuple[str, ...]
    val_speakers: tuple[str, ...]
    pad_frames: int
    fewshot_words: tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    hidden: int
    conv_layers: int
    conv_kernel: int
    gru_layers: int
    embedding_dim: int
    dropout: float
    logit_scale: float


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_epochs: int
    label_smoothing: float
    supcon_weight: float
    supcon_temperature: float
    patience: int
    max_frames: int
    min_frames: int
    time_stretch: tuple[float, float]
    feature_noise: float
    frame_drop: float
    num_workers: int


@dataclass(frozen=True)
class EvaluateConfig:
    category_gate: float
    fewshot_shots: int
    fewshot_episodes: int
    target_precision: float


@dataclass(frozen=True)
class ReplayConfig:
    words: tuple[str, ...]
    speakers: tuple[str, ...]
    context_frames: int
    crop_width: int
    crop_height: int


@dataclass(frozen=True)
class Config:
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    evaluate: EvaluateConfig
    replay: ReplayConfig

    @classmethod
    def load(cls, path: Path) -> Self:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
        extra = set(raw) - {"seed", "data", "model", "train", "evaluate", "replay"}
        if extra:
            raise ValueError(f"unknown config sections: {sorted(extra)}")
        cfg = cls(
            seed=int(raw["seed"]),
            data=_build(DataConfig, raw["data"]),
            model=_build(ModelConfig, raw["model"]),
            train=_build(TrainConfig, raw["train"]),
            evaluate=_build(EvaluateConfig, raw["evaluate"]),
            replay=_build(ReplayConfig, raw["replay"]),
        )
        cfg.validate()
        return cfg

    @property
    def trained_words(self) -> tuple[str, ...]:
        """Words the classifier learns, in logit order."""
        held = set(self.data.fewshot_words)
        return tuple(w for w in ALL_WORDS if w not in held)

    def validate(self) -> None:
        unknown = [w for w in (*self.data.fewshot_words, *self.replay.words) if w not in ALL_WORDS]
        if unknown:
            raise ValueError(f"not GRID words: {unknown}")
        overlap = set(self.data.test_speakers) & set(self.data.val_speakers)
        if overlap:
            raise ValueError(f"speakers in both test and val: {sorted(overlap)}")
        if set(self.replay.words) & set(self.data.fewshot_words):
            raise ValueError("replay words must be words the classifier was trained on")
        if not set(self.replay.speakers) <= set(self.data.test_speakers):
            raise ValueError("replay clips must come from held-out test speakers")
        for category, words in CATEGORIES.items():
            if all(w in self.data.fewshot_words for w in words):
                raise ValueError(f"few-shot words leave category {category!r} empty")
        if not 1 <= self.train.min_frames <= self.train.max_frames:
            raise ValueError("need 1 <= min_frames <= max_frames")


def _build[T](cls: type[T], values: dict[str, Any]) -> T:
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    extra = set(values) - names
    missing = names - set(values)
    if extra or missing:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(extra)}, missing {sorted(missing)}")
    converted = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
    return cls(**converted)
