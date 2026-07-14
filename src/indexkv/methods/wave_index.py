"""RetroInfer retrieve-zone index normalized to a strict token budget.

The build follows upstream segment_k_means: temporal-midpoint initialization,
segment-local normalized updates, then one global unnormalized assignment.
The pure-torch implementation uses chunked scores and scatter reductions, so it
never creates an H-by-N-by-C one-hot tensor. Decode ranks centroids with the
upstream group-summed softmax score and returns whole retrieve-zone clusters.
RetroInfer's aggregate estimation zone is not position-expressible and remains
outside this index-only comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..ops import (
    cluster_members_csr,
    middle_region,
    segmented_kmeans_ip,
    take_whole_clusters_under_budget,
)
from ..registry import IndexMethod, register
from ..types import LayerSelection, MethodConfig, QueryNeeds


@dataclass
class WaveIndexData:
    centroids: torch.Tensor
    members: torch.Tensor
    offsets: torch.Tensor
    sizes: torch.Tensor
    lo: int
    num_segments: int


def _cluster_layout(
    num_tokens: int,
    avg_cluster_size: int,
    requested_segments: int,
    explicit_clusters,
) -> tuple[int, int]:
    if num_tokens <= 0:
        return 0, 1
    if avg_cluster_size <= 0:
        raise ValueError(
            f"avg_cluster_size must be > 0, got {avg_cluster_size}"
        )
    if requested_segments <= 0:
        raise ValueError(
            f"n_segments must be > 0, got {requested_segments}"
        )
    segments = min(requested_segments, num_tokens)

    if explicit_clusters is not None:
        clusters = int(explicit_clusters)
        if clusters <= 0 or clusters > num_tokens:
            raise ValueError(
                f"n_clusters must be in [1, {num_tokens}], got {clusters}"
            )
        if clusters % segments:
            raise ValueError(
                f"n_clusters={clusters} must be divisible by n_segments={segments}"
            )
        return clusters, segments

    target = max(round(num_tokens / avg_cluster_size), 1)
    factor = math.lcm(8, segments)
    candidates = list(range(factor, num_tokens + 1, factor))
    if candidates:
        clusters = min(candidates, key=lambda value: (abs(value - target), value))
    else:
        max_multiple = (num_tokens // segments) * segments
        candidates = list(range(segments, max_multiple + 1, segments))
        clusters = min(
            candidates, key=lambda value: (abs(value - target), value)
        )
    return clusters, segments


@register(
    "wave_index",
    kind="per_head",
    needs=QueryNeeds(query="last"),
    reselect="per_step",
    reference="RetrievalAttention/RetroInfer segmented k-means cluster retrieval (retrieve zone only)",
)
class WaveIndex(IndexMethod):
    def build(self, K, V, cfg: MethodConfig, Q=None):
        H, N, D = K.shape
        lo, hi = middle_region(N, cfg.sink, cfg.recent)
        num_tokens = hi - lo
        if num_tokens == 0:
            return WaveIndexData(
                centroids=torch.empty(
                    H, 0, D, dtype=torch.float32, device=K.device
                ),
                members=torch.empty(
                    H, 0, dtype=torch.long, device=K.device
                ),
                offsets=torch.zeros(
                    H, 1, dtype=torch.long, device=K.device
                ),
                sizes=torch.empty(
                    H, 0, dtype=torch.long, device=K.device
                ),
                lo=lo,
                num_segments=1,
            )

        avg_cluster_size = int(cfg.get("avg_cluster_size", 16))
        default_segments = max(round(num_tokens / 8192), 1)
        requested_segments = int(
            cfg.get("n_segments", cfg.get("num_segments", default_segments))
        )
        num_clusters, num_segments = _cluster_layout(
            num_tokens,
            avg_cluster_size,
            requested_segments,
            cfg.get("n_clusters"),
        )
        n_iter = int(cfg.get("kmeans_iter", 10))
        assignment_chunk = int(cfg.get("assignment_chunk_size", 256))

        keys = K[:, lo:hi, :].float()
        mean_key = keys.mean(dim=1, keepdim=True)
        centered_keys = keys - mean_key
        centroids, labels = segmented_kmeans_ip(
            centered_keys,
            num_clusters,
            n_iter=n_iter,
            num_segments=num_segments,
            token_chunk_size=assignment_chunk,
        )
        centroids = centroids + mean_key
        members, offsets, sizes = cluster_members_csr(
            labels, num_clusters
        )
        return WaveIndexData(
            centroids=centroids,
            members=members + lo,
            offsets=offsets,
            sizes=sizes,
            lo=lo,
            num_segments=num_segments,
        )

    def select(
        self, index: WaveIndexData, Q, cfg: MethodConfig
    ) -> LayerSelection:
        H, num_clusters, D = index.centroids.shape
        if num_clusters == 0:
            indices = torch.full(
                (H, cfg.budget), -1, dtype=torch.long, device=Q.device
            )
            valid = torch.zeros(
                (H, cfg.budget), dtype=torch.bool, device=Q.device
            )
            return LayerSelection(
                kind="per_head",
                per_head_idx=indices,
                per_head_valid=valid,
            )

        grouped_query = Q[:, -1, :].reshape(
            H, cfg.group_size, D
        ).float()
        logits = torch.einsum(
            "hgd,hcd->hgc", grouped_query, index.centroids
        ) / math.sqrt(D)
        empty = index.sizes == 0
        logits.masked_fill_(empty[:, None, :], float("-inf"))
        scores = logits.softmax(dim=-1).sum(dim=1)
        scores.masked_fill_(empty, float("-inf"))
        order = scores.argsort(dim=-1, descending=True, stable=True)
        indices, valid = take_whole_clusters_under_budget(
            order,
            index.members,
            index.offsets,
            index.sizes,
            cfg.budget,
        )
        return LayerSelection(
            kind="per_head",
            per_head_idx=indices,
            per_head_valid=valid,
        )
