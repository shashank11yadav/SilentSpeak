"""The lip-motion word encoder and its contrastive loss.

``frames [B, T, FEATURE_DIM]`` -> temporal convolutions -> 2-layer BiGRU ->
attention pooling -> L2-normalised ``embedding [B, 128]``. The word logits are a
scaled cosine between the embedding and one learned vector per trained word,
so the classifier and the "Teach your own words" prototypes live in the same
space.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .config import ModelConfig
from .features import FEATURE_DIM


class ConvBlock(nn.Module):
    def __init__(self, hidden: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.conv = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        y = self.norm(x).transpose(1, 2)
        y = self.drop(F.gelu(self.conv(y))).transpose(1, 2)
        out: Tensor = (x + y) * mask
        return out


class LipEncoder(nn.Module):
    mean: Tensor
    std: Tensor

    def __init__(self, cfg: ModelConfig, n_classes: int, feature_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        h = cfg.hidden
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("std", torch.ones(feature_dim))
        self.inp = nn.Linear(feature_dim, h)
        self.convs = nn.ModuleList(
            ConvBlock(h, cfg.conv_kernel, cfg.dropout) for _ in range(cfg.conv_layers)
        )
        self.gru = nn.GRU(
            h,
            h,
            num_layers=cfg.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(2 * h, 1)
        self.proj = nn.Sequential(
            nn.Linear(2 * h, 2 * h), nn.GELU(), nn.Linear(2 * h, cfg.embedding_dim)
        )
        self.classes = nn.Parameter(torch.randn(n_classes, cfg.embedding_dim) * 0.1)
        self.logit_scale = cfg.logit_scale

    def set_normalisation(self, mean: Tensor, std: Tensor) -> None:
        self.mean.copy_(mean)
        self.std.copy_(std.clamp_min(1e-4))

    def forward(self, frames: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        b, t, _ = frames.shape
        if lengths is None:
            mask = torch.ones(b, t, 1, dtype=frames.dtype, device=frames.device)
        else:
            steps = torch.arange(t, device=frames.device)
            mask = (steps[None, :] < lengths.to(frames.device)[:, None]).to(frames.dtype)[..., None]

        x = (frames - self.mean) / self.std
        x = self.inp(x) * mask
        for block in self.convs:
            x = block(x, mask)

        if lengths is None:
            y, _ = self.gru(x)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.gru(packed)
            y, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=t)

        scores = self.attn(y)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * y).sum(dim=1)

        embedding = F.normalize(self.proj(pooled), dim=-1)
        logits = self.logit_scale * embedding @ F.normalize(self.classes, dim=-1).T
        return logits, embedding


def supcon_loss(embedding: Tensor, labels: Tensor, temperature: float) -> Tensor:
    """Supervised contrastive loss (Khosla et al., 2020) on L2-normalised embeddings.

    Anchors with no other sample of their class in the batch are skipped.
    """
    n = embedding.shape[0]
    sim = embedding @ embedding.T / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=embedding.device)
    sim = sim.masked_fill(self_mask, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positives = (labels[:, None] == labels[None, :]) & ~self_mask
    n_pos = positives.sum(dim=1)
    has_pos = n_pos > 0
    if not bool(has_pos.any()):
        return embedding.sum() * 0.0
    mean_log_prob = (log_prob * positives).sum(dim=1)[has_pos] / n_pos[has_pos]
    return -mean_log_prob.mean()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
