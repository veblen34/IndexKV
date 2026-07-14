"""HATA (Gong et al., ACL'25 Findings) — trainable hash-aware top-k, index only.

Per-KV-head linear hash projection ``K @ W`` (W: (H_kv, D, rbits), trained),
binarized by sign. At every decode step the query is hashed with the same W
(one code per query head) and each cached key is scored by Hamming distance
summed over the GQA group; the ``budget`` smallest-distance tokens are kept
per kv-head — targeting the reference's group-summed ``hamming_score`` +
``batch_topk(largest=False)`` (kvcache_hash.py:235-250).

Instead of packing bits and XOR+popcount we keep ±1 codes and use the identity
``hamming(a, b) = (rbits - a·b) / 2``.  This is an algebraically equivalent
score representation; the pure-torch port remains unverified against upstream
end-to-end behavior (the packed-word CUDA kernel is excluded system
acceleration).

The framework applies the reference's two dense prefix layers globally to every
method. Missing HATA weights on any remaining sparse layer fail fast.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..ops import middle_region
from ..registry import IndexMethod, register
from ..types import LayerSelection, MethodConfig, QueryNeeds


@dataclass
class HataIndex:
    codes: torch.Tensor    # (H_kv, N_mid, rbits) ±1 codes of middle-region keys
    W: torch.Tensor        # (H_kv, D, rbits) hash projection
    lo: int                # middle-region offset


@register(
    "hata",
    kind="per_head",
    needs=QueryNeeds(query="last", weights_key="hata"),
    reselect="per_step",
    reference="HATA (kvcache_hash.py): group-summed Hamming top-k over trained hash codes",
)
class Hata(IndexMethod):
    def build(self, K, V, cfg: MethodConfig, Q=None):
        W = cfg.get("weights")
        if W is None:
            raise ValueError(
                f"hata weights are missing for sparse layer {cfg.layer_idx}"
            )
        H, N, D = K.shape
        lo, hi = middle_region(N, cfg.sink, cfg.recent)
        proj = torch.einsum("hnd,hdr->hnr", K[:, lo:hi, :].float(), W.float())
        codes = torch.where(proj > 0, 1.0, -1.0).to(torch.bfloat16)
        return HataIndex(codes=codes, W=W, lo=lo)

    def select(self, index: HataIndex, Q, cfg: MethodConfig) -> LayerSelection:
        H, N_mid, R = index.codes.shape
        Qg = Q[:, -1, :].reshape(H, cfg.group_size, -1)
        qproj = torch.einsum("hgd,hdr->hgr", Qg.float(), index.W.float())
        qcodes = torch.where(qproj > 0, 1.0, -1.0).to(index.codes.dtype)
        # sum of ±1 inner products over the GQA group == g*rbits - 2*sum(hamming)
        score = torch.einsum(
            "hgr,hnr->hn", qcodes.float(), index.codes.float()
        )
        kb = min(cfg.budget, N_mid)
        idx = score.topk(kb, dim=-1).indices + index.lo   # (H_kv, kb)
        return LayerSelection(kind="per_head", per_head_idx=idx)
