"""Shared pure-torch operations used by index implementations.

These helpers preserve the common selectable middle region and represent
ragged outputs explicitly instead of padding them with real prompt positions.
"""

from __future__ import annotations

from typing import Tuple

import torch


def middle_region(n: int, sink: int, recent: int) -> Tuple[int, int]:
    """Return the common selectable half-open interval."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if sink < 0 or recent < 0:
        raise ValueError(
            f"sink/recent must be >= 0, got sink={sink}, recent={recent}"
        )
    lo = min(sink, n)
    hi = max(lo, n - recent)
    return lo, hi


@torch.no_grad()
def _assign_ip_chunked(
    X: torch.Tensor,
    centroids: torch.Tensor,
    *,
    token_chunk_size: int,
) -> torch.Tensor:
    """Assign by inner product without materializing a full (B, N, C)."""
    B, N, _ = X.shape
    C = centroids.shape[1]
    if C == 0:
        raise ValueError("at least one centroid is required")
    step = max(1, int(token_chunk_size))
    labels = torch.empty(B, N, dtype=torch.long, device=X.device)
    centroids_t = centroids.transpose(1, 2)
    for start in range(0, N, step):
        end = min(start + step, N)
        labels[:, start:end] = torch.bmm(
            X[:, start:end], centroids_t
        ).argmax(dim=-1)
    return labels


@torch.no_grad()
def _update_centroids(
    X: torch.Tensor,
    labels: torch.Tensor,
    previous: torch.Tensor,
    *,
    normalize: bool,
    token_chunk_size: int,
) -> torch.Tensor:
    """Scatter-reduce means without constructing one-hot assignments."""
    B, N, D = X.shape
    C = previous.shape[1]
    sums = torch.zeros(B, C, D, dtype=torch.float32, device=X.device)
    counts = torch.zeros(B, C, dtype=torch.float32, device=X.device)
    step = max(1, int(token_chunk_size))
    for start in range(0, N, step):
        end = min(start + step, N)
        lab = labels[:, start:end]
        sums.scatter_add_(
            1,
            lab.unsqueeze(-1).expand(-1, -1, D),
            X[:, start:end].float(),
        )
        counts.scatter_add_(
            1, lab, torch.ones_like(lab, dtype=torch.float32)
        )

    means = sums / counts.clamp_min(1.0).unsqueeze(-1)
    if normalize:
        means = means / means.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.where(
        (counts > 0).unsqueeze(-1), means, previous.float()
    )


@torch.no_grad()
def segmented_kmeans_ip(
    X: torch.Tensor,
    n_clusters: int,
    n_iter: int = 10,
    num_segments: int = 1,
    token_chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch port of RetroInfer segment_k_means.

    Initialization uses uniform temporal midpoints. The first n_iter - 1
    updates are segment-local and normalize nonempty centroids. The final
    update assigns all tokens globally and stores unnormalized means.
    """
    if X.ndim != 3:
        raise ValueError(
            f"X must have shape (B, N, D), got {tuple(X.shape)}"
        )
    B, N, D = X.shape
    if N == 0:
        raise ValueError("cannot cluster an empty token sequence")
    C = int(n_clusters)
    S = int(num_segments)
    if C <= 0 or C > N:
        raise ValueError(f"n_clusters must be in [1, {N}], got {C}")
    if S <= 0 or S > N:
        raise ValueError(f"num_segments must be in [1, {N}], got {S}")
    if C % S:
        raise ValueError(
            f"n_clusters={C} must be divisible by num_segments={S}"
        )
    if n_iter <= 0:
        raise ValueError(f"n_iter must be > 0, got {n_iter}")

    Xf = X.float()
    midpoint = (
        (torch.arange(C, device=X.device, dtype=torch.float32) + 0.5)
        * (N / C)
    ).to(torch.long).clamp_max(N - 1)
    centroids = Xf.index_select(1, midpoint).clone()

    tokens_per_segment = N // S
    used_tokens = tokens_per_segment * S
    centroids_per_segment = C // S
    segmented_data = Xf[:, :used_tokens].reshape(
        B * S, tokens_per_segment, D
    )
    segmented_centroids = centroids.reshape(
        B * S, centroids_per_segment, D
    )
    for _ in range(n_iter - 1):
        segmented_labels = _assign_ip_chunked(
            segmented_data,
            segmented_centroids,
            token_chunk_size=token_chunk_size,
        )
        segmented_centroids = _update_centroids(
            segmented_data,
            segmented_labels,
            segmented_centroids,
            normalize=True,
            token_chunk_size=token_chunk_size,
        )

    centroids = segmented_centroids.reshape(B, C, D)
    labels = _assign_ip_chunked(
        Xf, centroids, token_chunk_size=token_chunk_size
    )
    centroids = _update_centroids(
        Xf,
        labels,
        centroids,
        normalize=False,
        token_chunk_size=token_chunk_size,
    )
    return centroids, labels


@torch.no_grad()
def cluster_members_csr(
    labels: torch.Tensor, n_clusters: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a compact per-batch CSR representation of cluster members."""
    B, _ = labels.shape
    order = labels.argsort(dim=1, stable=True)
    sizes = torch.zeros(
        B, n_clusters, dtype=torch.long, device=labels.device
    )
    sizes.scatter_add_(1, labels, torch.ones_like(labels))
    offsets = torch.zeros(
        B, n_clusters + 1, dtype=torch.long, device=labels.device
    )
    offsets[:, 1:] = sizes.cumsum(dim=1)
    return order, offsets, sizes


@torch.no_grad()
def take_whole_clusters_under_budget(
    order: torch.Tensor,
    members: torch.Tensor,
    offsets: torch.Tensor,
    sizes: torch.Tensor,
    budget: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select whole ranked clusters without exceeding a token budget.

    Clusters too large for the remaining capacity are skipped, not truncated.
    Invalid output slots contain -1 and are identified by the returned mask.
    """
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    B, _ = order.shape
    out = torch.full(
        (B, budget), -1, dtype=torch.long, device=order.device
    )
    valid = torch.zeros(
        (B, budget), dtype=torch.bool, device=order.device
    )
    for b in range(B):
        used = 0
        for cluster_id in order[b].tolist():
            count = int(sizes[b, cluster_id])
            if count == 0:
                continue
            if count > budget - used:
                continue
            start = int(offsets[b, cluster_id])
            end = int(offsets[b, cluster_id + 1])
            out[b, used:used + count] = members[b, start:end]
            valid[b, used:used + count] = True
            used += count
            if used == budget:
                break
    return out, valid
