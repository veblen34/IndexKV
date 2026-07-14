"""Louver (UIC-InDeXLab/Louver) — reservoir-sample-estimated halfspace threshold.

Faithful to the accuracy path the Louver evals actually run
(``benchmark_area/experiments/accuracy/louver_hf/``): at prefill a random
reservoir of ``sample_size`` keys is drawn per kv-head; at each decode step the
raw query scores that sample, a per-query-head threshold ``tau`` is estimated
from the sample (budget mode: the ``k``-th largest sample score for
``k = int(budget_fraction * M)``), and every prompt key with ``q.k >= tau`` is
kept (``louver_hf/threshold.py:49-140``, ``attention.py`` TA filter). The count
is data-dependent, NOT a fixed top-k: Louver's ball tree only guarantees zero
false negatives *relative to the estimated tau*, so the selected set differs
from the exact per-head top-k. The ball tree / triton filter are excluded
efficiency machinery; the selected SET is reproduced here in pure torch.

To fit the framework's fixed per-head budget, the threshold-passing set is
capped to the common budget (top-``budget`` by score among those clearing the
SAMPLED tau) — the cut is always the sampled tau, never the exact ``budget``-th
order statistic of the full scores (that would re-introduce the top-k oracle).

Knobs: ``sample_size`` (reservoir size M, default 256), ``budget_fraction``
(defaults to ``budget / N_mid`` so the expected retrieved count matches budget).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..ops import middle_region
from ..registry import IndexMethod, register
from ..types import LayerSelection, MethodConfig, QueryNeeds


@dataclass
class RangeIndex:
    K_mid: torch.Tensor    # (H_kv, N_mid, D) middle-region keys
    sample: torch.Tensor   # (H_kv, M, D) reservoir sample of middle-region keys
    lo: int


@register(
    "range_search",
    kind="per_head",
    needs=QueryNeeds(query="last"),
    reselect="per_step",
    reference="Louver (louver_hf): reservoir-sample-estimated halfspace threshold, per-query-head",
)
class RangeSearch(IndexMethod):
    def build(self, K, V, cfg: MethodConfig, Q=None):
        H, N, D = K.shape
        lo, hi = middle_region(N, cfg.sink, cfg.recent)
        K_mid = K[:, lo:hi, :]
        N_mid = K_mid.shape[1]
        M = min(int(cfg.get("sample_size", 256)), N_mid)
        if M > 0:
            # deterministic per-layer reservoir draw (reproducible random subset)
            gen = torch.Generator(device=K.device).manual_seed(0x10DE + cfg.layer_idx)
            idx = torch.randperm(N_mid, device=K.device, generator=gen)[:M]
            sample = K_mid[:, idx, :]
        else:
            sample = K_mid[:, :0, :]
        return RangeIndex(K_mid=K_mid, sample=sample, lo=lo)

    def select(self, index: RangeIndex, Q, cfg: MethodConfig) -> LayerSelection:
        H_kv, N_mid, D = index.K_mid.shape
        g = cfg.group_size
        H_q = H_kv * g
        if N_mid == 0:
            return LayerSelection(
                kind="per_head",
                per_head_idx=torch.empty(H_q, 0, dtype=torch.long, device=Q.device),
            )
        Qg = Q[:, -1, :].reshape(H_kv, g, D).float()
        M = index.sample.shape[1]
        # (a) sampled threshold tau per query head (budget mode)
        s_scores = torch.einsum(
            "hgd,hmd->hgm", Qg, index.sample.float()
        ).reshape(H_q, M)                                          # (H_q, M)
        frac = cfg.get("budget_fraction", None)
        frac = (cfg.budget / N_mid) if frac is None else float(frac)
        k = max(1, min(M, int(frac * M)))
        tau = s_scores.topk(k, dim=-1).values[:, -1]              # (H_q,)
        # (b) keep every middle key with q.k >= tau (halfspace); cap to budget
        full = torch.einsum(
            "hgd,hnd->hgn", Qg, index.K_mid.float()
        ).reshape(H_q, N_mid)                                      # (H_q, N_mid)
        passes = full >= tau.unsqueeze(1)
        masked = full.masked_fill(~passes, float("-inf"))
        kb = min(cfg.budget, N_mid)
        top_vals, top_idx = masked.topk(kb, dim=-1)               # (H_q, kb)
        valid = top_vals > float("-inf")                          # only true threshold-passers
        per_head_idx = torch.where(
            valid, top_idx + index.lo, top_idx.new_full((), -1)
        )
        return LayerSelection(
            kind="per_head", per_head_idx=per_head_idx, per_head_valid=valid
        )
