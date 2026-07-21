"""ChunkKV over SnapKV scoring, normalized to the index-only contract.

The last observation-window prefill queries produce SnapKV token scores once
while the budget-independent index is built. Scores are averaged into semantic
chunks wholly inside the common selectable middle region. A budget-specific
selection then takes ranked whole chunks without crossing the shared token
budget; sink and recent positions are added only by the common mask builder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..ops import middle_region
from ..registry import IndexMethod, register
from ..types import LayerSelection, MethodConfig, QueryNeeds


@dataclass
class ChunkKVIndex:
    blocks: torch.Tensor       # (num_chunks, 2), absolute [start, end)
    scores: torch.Tensor       # (num_chunks,), budget-independent scores
    order: torch.Tensor        # (num_chunks,), descending score order
    chunk_length: int


@register(
    "chunkkv",
    kind="block",
    needs=QueryNeeds(query="obs_window", obs_window=64),
    reselect="static",
    reference="kvpress ChunkKV over SnapKV scoring (static prefill-time chunk ranking)",
)
class ChunkKV(IndexMethod):
    def build(self, K, V, cfg: MethodConfig, Q=None):
        if Q is None:
            raise ValueError("chunkkv requires its captured observation-window Q at build")
        H_kv, N, D = K.shape
        H_q, W, q_dim = Q.shape
        if q_dim != D or H_q != H_kv * cfg.group_size:
            raise ValueError(
                "chunkkv Q/K dimensions do not match: "
                f"Q={tuple(Q.shape)}, K={tuple(K.shape)}, "
                f"group_size={cfg.group_size}"
            )
        if W <= 0 or W > N:
            raise ValueError(
                f"chunkkv observation window must be in [1, {N}], got {W}"
            )

        chunk_length = int(cfg.get("chunk_length", 20))
        kernel_size = int(cfg.get("kernel_size", 5))
        if chunk_length <= 0:
            raise ValueError(
                f"chunk_length must be > 0, got {chunk_length}"
            )
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )

        lo, hi = middle_region(N, cfg.sink, cfg.recent)
        starts = torch.arange(lo, hi, chunk_length, device=K.device)
        ends = (starts + chunk_length).clamp(max=hi)
        blocks = (
            torch.stack((starts, ends), dim=-1).to(torch.long)
            if starts.numel()
            else torch.empty(0, 2, dtype=torch.long, device=K.device)
        )
        if blocks.shape[0] == 0:
            empty_scores = torch.empty(0, dtype=torch.float32, device=K.device)
            return ChunkKVIndex(
                blocks=blocks,
                scores=empty_scores,
                order=torch.empty(0, dtype=torch.long, device=K.device),
                chunk_length=chunk_length,
            )

        # Match kvpress SnapKV numerical semantics: QK is evaluated in the
        # model/capture dtype, softmax accumulates in FP32, and its result is
        # cast back to the query dtype before pooling and chunk aggregation.
        grouped_queries = Q.reshape(
            H_kv, cfg.group_size, W, D
        )
        logits = torch.einsum(
            "hgtd,hnd->hgtn", grouped_queries, K
        ).reshape(H_q, W, N) / math.sqrt(D)
        query_row = torch.arange(W, device=K.device).view(-1, 1)
        key_col = torch.arange(N, device=K.device).view(1, -1)
        logits.masked_fill_(
            key_col > query_row + (N - W), float("-inf")
        )
        attention = logits.softmax(
            dim=-1, dtype=torch.float32
        ).to(grouped_queries.dtype)

        prefix_len = N - W
        if prefix_len:
            token_scores = attention[..., :prefix_len].mean(dim=-2)
            token_scores = F.avg_pool1d(
                token_scores.unsqueeze(0),
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
            ).squeeze(0)
            token_scores = token_scores.view(
                H_kv, cfg.group_size, prefix_len
            ).mean(dim=1)
            high_score = token_scores.max() + 1.0
        else:
            token_scores = torch.empty(
                H_kv, 0, dtype=grouped_queries.dtype, device=K.device
            )
            high_score = torch.tensor(
                1.0, dtype=grouped_queries.dtype, device=K.device
            )

        all_scores = torch.empty(
            H_kv, N, dtype=grouped_queries.dtype, device=K.device
        )
        all_scores[:, :prefix_len] = token_scores
        all_scores[:, prefix_len:] = high_score
        global_scores = all_scores.sum(dim=0)

        # Use the same direct per-chunk mean as kvpress. A full-sequence BF16
        # prefix sum followed by subtraction both changes reduction order and
        # loses substantial precision for late chunks.
        eligible_scores = global_scores[lo:hi]
        num_complete = eligible_scores.numel() // chunk_length
        complete_end = num_complete * chunk_length
        chunk_parts = []
        if num_complete:
            chunk_parts.append(
                eligible_scores[:complete_end]
                .reshape(num_complete, chunk_length)
                .mean(dim=-1)
            )
        if complete_end < eligible_scores.numel():
            chunk_parts.append(
                eligible_scores[complete_end:].mean().reshape(1)
            )
        chunk_scores = torch.cat(chunk_parts)
        order = chunk_scores.argsort(descending=True, stable=True)
        return ChunkKVIndex(
            blocks=blocks,
            scores=chunk_scores,
            order=order,
            chunk_length=chunk_length,
        )

    def select(self, index: ChunkKVIndex, Q, cfg: MethodConfig) -> LayerSelection:
        if Q is not None:
            raise ValueError("static chunkkv selection must not receive a live query")
        if index.blocks.shape[0] == 0:
            return LayerSelection(
                kind="block", blocks=index.blocks.new_empty((0, 2))
            )

        selected = []
        used = 0
        for chunk_id in index.order.tolist():
            block = index.blocks[chunk_id]
            size = int(block[1] - block[0])
            if size > cfg.budget - used:
                continue
            selected.append(block)
            used += size
            if used == cfg.budget:
                break
        blocks = (
            torch.stack(selected, dim=0)
            if selected
            else index.blocks.new_empty((0, 2))
        )
        return LayerSelection(kind="block", blocks=blocks)
