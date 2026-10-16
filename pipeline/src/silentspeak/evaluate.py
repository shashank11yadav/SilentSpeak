"""Held-out-speaker and few-shot evaluation, "not sure" calibration and the category gate."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import Config
from .dataset import Segments, WordDataset, split_mask, word_labels
from .train import embed, load_checkpoint, pick_device
from .vocab import CATEGORIES, WORD_CATEGORY

F32 = NDArray[np.float32]


def softmax(x: F32) -> F32:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def calibrate_threshold(
    scores: NDArray[np.float64], correct: NDArray[np.bool_], target: float, min_accepted: int = 20
) -> float:
    """Smallest threshold whose accepted answers (score >= t) are right at least ``target`` often.

    If no threshold reaches the target with at least ``min_accepted`` answers,
    the highest score is returned, so almost everything becomes "not sure".
    """
    if len(scores) == 0:
        return 1.0
    order = np.argsort(-scores)
    s, c = scores[order], correct[order].astype(np.float64)
    precision = np.cumsum(c) / np.arange(1, len(c) + 1)
    need = min(min_accepted, len(c))
    # Walk from the most to the least confident answer; the last cut that still
    # meets the target gives the lowest usable threshold.
    ok = np.flatnonzero((precision >= target) & (np.arange(1, len(c) + 1) >= need))
    if len(ok) == 0:
        return float(s[0])
    return float(s[ok[-1]])


def accepted_stats(
    scores: NDArray[np.float64], correct: NDArray[np.bool_], threshold: float
) -> tuple[float | None, float]:
    """(precision, coverage) of answers at or above the threshold."""
    keep = scores >= threshold
    coverage = float(keep.mean()) if len(keep) else 0.0
    precision = float(correct[keep].mean()) if keep.any() else None
    return precision, coverage


def category_scores(
    logits: F32, labels: NDArray[np.int64], words: tuple[str, ...], category: str
) -> tuple[NDArray[np.int64], NDArray[np.int64], F32]:
    """Rows of this category, label positions within it, and in-category probabilities."""
    idx = np.array([i for i, w in enumerate(words) if WORD_CATEGORY[w] == category])
    rows = np.flatnonzero(np.isin(labels, idx))
    probs = softmax(logits[rows][:, idx])
    target = np.searchsorted(idx, labels[rows])
    return rows, target.astype(np.int64), probs


def fewshot_episodes(
    emb: F32,
    words: NDArray[np.str_],
    speakers: NDArray[np.str_],
    vocab: tuple[str, ...],
    shots: int,
    episodes: int,
    rng: np.random.Generator,
    queries_per_word: int = 5,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """N-way K-shot nearest-prototype accuracy within one speaker per episode.

    Returns the accuracy plus, for every query, the best cosine similarity,
    the margin over the runner-up and whether it was right.
    """
    by_speaker: dict[str, dict[str, NDArray[np.int64]]] = {}
    for spk in sorted(set(speakers.tolist())):
        per_word = {w: np.flatnonzero((speakers == spk) & (words == w)) for w in vocab}
        if all(len(ix) > shots for ix in per_word.values()):
            by_speaker[spk] = per_word
    if not by_speaker:
        return float("nan"), np.zeros(0), np.zeros(0), np.zeros(0, bool)
    names = sorted(by_speaker)
    tops, margins, rights = [], [], []
    for _ in range(episodes):
        per_word = by_speaker[names[int(rng.integers(len(names)))]]
        protos, queries, truth = [], [], []
        for k, w in enumerate(vocab):
            ix = rng.permutation(per_word[w])
            p = emb[ix[:shots]].mean(axis=0)
            protos.append(p / max(float(np.linalg.norm(p)), 1e-8))
            q = ix[shots : shots + queries_per_word]
            queries.extend(q.tolist())
            truth.extend([k] * len(q))
        sims = emb[queries] @ np.stack(protos).T
        ranked = np.sort(sims, axis=1)
        tops.append(ranked[:, -1])
        margins.append(ranked[:, -1] - ranked[:, -2])
        rights.append(sims.argmax(axis=1) == np.array(truth))
    right = np.concatenate(rights)
    return (
        float(right.mean()),
        np.concatenate(tops).astype(np.float64),
        np.concatenate(margins).astype(np.float64),
        right,
    )


def evaluate(
    segments: Segments, cfg: Config, checkpoint: Path, device_name: str = "cpu"
) -> dict[str, Any]:
    device = pick_device(device_name)
    model, words_list = load_checkpoint(checkpoint, cfg)
    model.to(device)
    words = tuple(words_list)
    labels = word_labels(segments, words)
    rng = np.random.default_rng(cfg.seed)
    ec = cfg.evaluate

    def split_outputs(split: str) -> tuple[Segments, NDArray[np.int64], F32, F32]:
        mask = split_mask(segments, cfg, split)  # type: ignore[arg-type]
        seg = segments.subset(mask)
        lab = labels[mask]
        ds = WordDataset(seg, np.maximum(lab, 0), cfg, train=False)
        logits, emb = embed(model, ds, device)
        return seg, lab, logits, emb

    test_seg, test_labels, test_logits, test_emb = split_outputs("test")
    val_seg, val_labels, val_logits, val_emb = split_outputs("val")

    categories = []
    for cat, all_words in CATEGORIES.items():
        cat_words = [w for w in all_words if w in words]
        rows, target, probs = category_scores(test_logits, test_labels, words, cat)
        vrows, vtarget, vprobs = category_scores(val_logits, val_labels, words, cat)
        n_words = len(cat_words)
        k = min(3, n_words)
        if len(rows):
            top1 = float((probs.argmax(1) == target).mean())
            topk = np.argsort(-probs, axis=1)[:, :k]
            top3 = float((topk == target[:, None]).any(axis=1).mean())
        else:
            top1 = top3 = float("nan")
        threshold = calibrate_threshold(
            vprobs.max(1).astype(np.float64) if len(vrows) else np.zeros(0),
            (vprobs.argmax(1) == vtarget) if len(vrows) else np.zeros(0, bool),
            ec.target_precision,
        )
        precision, coverage = (
            accepted_stats(probs.max(1).astype(np.float64), probs.argmax(1) == target, threshold)
            if len(rows)
            else (None, 0.0)
        )
        confusions = Counter(
            (cat_words[t], cat_words[p])
            for t, p in zip(target.tolist(), probs.argmax(1).tolist(), strict=True)
            if t != p
        )
        categories.append(
            {
                "id": cat,
                "words": cat_words,
                "n": len(rows),
                "top1": top1,
                "top3": top3,
                "chance1": 1 / n_words,
                "chance3": k / n_words,
                "threshold": threshold,
                "precisionAtThreshold": precision,
                "coverageAtThreshold": coverage,
                "shipped": bool(len(rows) and top1 >= ec.category_gate),
                "confusions": [
                    {"true": t, "predicted": p, "count": c}
                    for (t, p), c in confusions.most_common(5)
                ],
            }
        )

    known = test_labels >= 0
    overall_top1 = float((test_logits[known].argmax(1) == test_labels[known]).mean())
    top5 = np.argsort(-test_logits[known], axis=1)[:, :5]
    overall_top5 = float((top5 == test_labels[known][:, None]).any(axis=1).mean())

    fs_words = cfg.data.fewshot_words
    fs_acc, _, _, _ = fewshot_episodes(
        test_emb,
        test_seg.words,
        test_seg.speakers,
        fs_words,
        ec.fewshot_shots,
        ec.fewshot_episodes,
        rng,
    )
    _, vtop, vmargin, vright = fewshot_episodes(
        val_emb,
        val_seg.words,
        val_seg.speakers,
        fs_words,
        ec.fewshot_shots,
        ec.fewshot_episodes,
        rng,
    )
    margin = calibrate_threshold(vmargin, vright, ec.target_precision)
    min_sim = float(np.quantile(vtop[vright], 0.05)) if vright.any() else 0.0
    _, ttop, tmargin, tright = fewshot_episodes(
        test_emb,
        test_seg.words,
        test_seg.speakers,
        fs_words,
        ec.fewshot_shots,
        ec.fewshot_episodes,
        rng,
    )
    keep = (tmargin >= margin) & (ttop >= min_sim)
    fs_precision = float(tright[keep].mean()) if keep.any() else None

    def speakers_of(split: str) -> list[str]:
        mask = split_mask(segments, cfg, split)  # type: ignore[arg-type]
        return sorted(set(segments.speakers[mask].tolist()), key=lambda s: int(s[1:]))

    return {
        "featureVersion": 1,
        "seed": cfg.seed,
        "speakers": {s: speakers_of(s) for s in ("train", "val", "test")},
        "segments": {
            "train": int((split_mask(segments, cfg, "train") & (labels >= 0)).sum()),
            "val": len(val_seg),
            "test": len(test_seg),
        },
        "gate": ec.category_gate,
        "targetPrecision": ec.target_precision,
        "categories": categories,
        "overall": {"top1": overall_top1, "top5": overall_top5, "n": int(known.sum())},
        "fewshot": {
            "words": list(fs_words),
            "shots": ec.fewshot_shots,
            "episodes": ec.fewshot_episodes,
            "accuracy": fs_acc,
            "chance": 1 / len(fs_words),
            "margin": margin,
            "minSimilarity": min_sim,
            "precisionAtThreshold": fs_precision,
            "coverageAtThreshold": float(keep.mean()) if len(keep) else 0.0,
        },
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:.1f}%"


def metrics_markdown(m: dict[str, Any]) -> str:
    """The README table. Every number comes from ``metrics.json``."""
    lines = [
        f"Held-out test speakers: {', '.join(m['speakers']['test'])} "
        f"({m['segments']['test']} word segments). "
        f"A category ships when its top-1 is at least {100 * m['gate']:.0f}%.",
        "",
        "| Category | Words | Top-1 | Top-3 | Chance (top-1) | Ships |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for c in m["categories"]:
        lines.append(
            f"| {c['id']} | {len(c['words'])} | {_pct(c['top1'])} | {_pct(c['top3'])} "
            f"| {_pct(c['chance1'])} | {'yes' if c['shipped'] else 'no'} |"
        )
    fs = m["fewshot"]
    lines += [
        "",
        f"All {m['overall']['n']} test segments, 45-way: top-1 {_pct(m['overall']['top1'])}, "
        f"top-5 {_pct(m['overall']['top5'])}.",
        "",
        f"Few-shot, words never trained on ({', '.join(fs['words'])}): "
        f"{len(fs['words'])}-way {fs['shots']}-shot accuracy {_pct(fs['accuracy'])} "
        f"over {fs['episodes']} episodes (chance {_pct(fs['chance'])}).",
        "",
    ]
    return "\n".join(lines)


def write_reports(metrics: dict[str, Any], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (reports_dir / "metrics.md").write_text(metrics_markdown(metrics))
