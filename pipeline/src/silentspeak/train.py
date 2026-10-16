"""Train the lip-motion encoder on training speakers, early-stopping on validation speakers."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from .config import Config, ModelConfig
from .dataset import (
    Segments,
    WordDataset,
    collate,
    feature_stats,
    split_mask,
    word_labels,
)
from .model import LipEncoder, supcon_loss
from .vocab import WORD_CATEGORY

log = logging.getLogger(__name__)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - also seeds libraries that use the global state
    torch.manual_seed(seed)


@torch.no_grad()
def embed(
    model: LipEncoder, dataset: WordDataset, device: torch.device, batch_size: int = 256
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.float32]]]:
    """Logits and embeddings for every item of an (unaugmented) dataset."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    logits, embs = [], []
    for x, lengths, _ in loader:
        lg, em = model(x.to(device), lengths)
        logits.append(lg.cpu().numpy())
        embs.append(em.cpu().numpy())
    if not logits:
        return np.zeros((0, len(model.classes)), np.float32), np.zeros((0, 1), np.float32)
    return np.concatenate(logits), np.concatenate(embs)


def category_top1(
    logits: np.ndarray[Any, np.dtype[np.float32]],
    labels: np.ndarray[Any, np.dtype[np.int64]],
    words: tuple[str, ...],
) -> dict[str, float]:
    """Top-1 accuracy per category, choosing only among that category's trained words."""
    out: dict[str, float] = {}
    cats = np.array([WORD_CATEGORY[w] for w in words])
    for cat in dict.fromkeys(cats):
        idx = np.flatnonzero(cats == cat)
        rows = np.isin(labels, idx)
        if not rows.any():
            continue
        pred = idx[np.argmax(logits[rows][:, idx], axis=1)]
        out[str(cat)] = float((pred == labels[rows]).mean())
    return out


def train(
    segments: Segments,
    cfg: Config,
    out_dir: Path,
    device_name: str = "auto",
    max_steps: int | None = None,
) -> Path:
    """Train and write ``best.pt`` plus ``train_log.json`` to ``out_dir``."""
    seed_everything(cfg.seed)
    device = pick_device(device_name)
    words = cfg.trained_words
    labels = word_labels(segments, words)
    trainable = labels >= 0
    train_mask = split_mask(segments, cfg, "train") & trainable
    val_mask = split_mask(segments, cfg, "val") & trainable
    train_seg, val_seg = segments.subset(train_mask), segments.subset(val_mask)
    if len(train_seg) == 0 or len(val_seg) == 0:
        raise ValueError("need training and validation segments; check the speaker split")
    train_labels, val_labels = labels[train_mask], labels[val_mask]
    log.info("train %d segments, val %d, device %s", len(train_seg), len(val_seg), device)

    model = LipEncoder(cfg.model, n_classes=len(words))
    model.set_normalisation(*feature_stats(train_seg, cfg))
    model.to(device)

    tc = cfg.train
    train_ds = WordDataset(train_seg, train_labels, cfg, train=True, seed=cfg.seed)
    val_ds = WordDataset(val_seg, val_labels, cfg, train=False)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=tc.batch_size,
        shuffle=True,
        drop_last=len(train_ds) > tc.batch_size,
        collate_fn=collate,
        num_workers=tc.num_workers,
        generator=generator,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    steps_per_epoch = max(1, len(loader))
    total = steps_per_epoch * tc.epochs
    warmup = steps_per_epoch * tc.warmup_epochs

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    history: list[dict[str, float]] = []
    best_score, bad_epochs, step = -1.0, 0, 0
    started = time.monotonic()
    for epoch in range(tc.epochs):
        model.train()
        losses = []
        for x, lengths, y in loader:
            x, y = x.to(device), y.to(device)
            logits, emb = model(x, lengths)
            ce = F.cross_entropy(logits, y, label_smoothing=tc.label_smoothing)
            con = supcon_loss(emb, y, tc.supcon_temperature)
            loss = ce + tc.supcon_weight * con
            opt.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        val_logits, _ = embed(model, val_ds, device)
        per_cat = category_top1(val_logits, val_labels, words)
        score = float(np.mean(list(per_cat.values())))
        overall = float((val_logits.argmax(1) == val_labels).mean())
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "val_category_top1": score,
            "val_top1_all": overall,
            "seconds": time.monotonic() - started,
        }
        history.append(row)
        log.info(
            "epoch %d loss %.4f val category top-1 %.3f (all %.3f)",
            epoch + 1,
            row["loss"],
            score,
            overall,
        )
        if score > best_score:
            best_score, bad_epochs = score, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model": asdict(cfg.model),
                    "words": list(words),
                    "epoch": epoch + 1,
                    "val_category_top1": score,
                    "seed": cfg.seed,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= tc.patience:
                log.info("early stop after %d epochs without improvement", bad_epochs)
                break
        if max_steps is not None and step >= max_steps:
            break
    (out_dir / "train_log.json").write_text(json.dumps(history, indent=2) + "\n")
    return best_path


def load_checkpoint(path: Path, cfg: Config) -> tuple[LipEncoder, list[str]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    words = [str(w) for w in ckpt["words"]]
    if words != list(cfg.trained_words):
        raise ValueError(
            "checkpoint words differ from the config; was it trained with another one?"
        )
    model = LipEncoder(ModelConfig(**ckpt["model"]), n_classes=len(words))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), words
