"""Numerical-fidelity checks against kvpress SnapKV + ChunkKV semantics."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indexkv.methods.chunkkv import ChunkKV  # noqa: E402
from indexkv.types import MethodConfig  # noqa: E402


def _kvpress_reference_scores(K, Q, *, group_size, chunk_length, kernel_size):
    """Direct transcription of SnapKVPress.score + ChunkKVPress scoring."""
    H_kv, N, D = K.shape
    H_q, W, _ = Q.shape
    repeated_keys = K.repeat_interleave(group_size, dim=0)
    logits = torch.matmul(Q, repeated_keys.transpose(1, 2)) / math.sqrt(D)
    mask = torch.ones_like(logits) * float("-inf")
    mask = torch.triu(mask, diagonal=N - W + 1)
    attention = F.softmax(
        logits + mask, dim=-1, dtype=torch.float32
    ).to(Q.dtype)[..., :-W]

    scores = F.avg_pool1d(
        attention.mean(dim=-2).unsqueeze(0),
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        stride=1,
    ).squeeze(0)
    scores = scores.view(H_kv, group_size, N - W).mean(dim=1)
    scores = F.pad(scores, (0, W), value=scores.max().item() + 1)
    global_scores = scores.sum(dim=0)
    return global_scores.reshape(-1, chunk_length).mean(dim=-1)


def test_chunkkv_preserves_official_score_dtype_and_reduction_order() -> None:
    torch.manual_seed(20260720)
    H_kv, group_size, N, D, W = 2, 2, 12, 8, 3
    chunk_length, kernel_size = 4, 3
    # float64 makes an accidental .float() conversion observable on every CPU.
    K = torch.randn(H_kv, N, D, dtype=torch.float64)
    Q = torch.randn(H_kv * group_size, W, D, dtype=torch.float64)
    cfg = MethodConfig(
        budget=8,
        sink=0,
        recent=0,
        group_size=group_size,
        n_prompt=N,
        device="cpu",
        dtype=torch.float64,
        extra={
            "chunk_length": chunk_length,
            "kernel_size": kernel_size,
        },
    )

    index = ChunkKV().build(K, None, cfg, Q=Q)
    expected = _kvpress_reference_scores(
        K,
        Q,
        group_size=group_size,
        chunk_length=chunk_length,
        kernel_size=kernel_size,
    )

    assert index.scores.dtype == Q.dtype
    torch.testing.assert_close(index.scores, expected, rtol=1e-12, atol=1e-12)
