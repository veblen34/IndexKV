"""Self-Indexing KVCache — the repo's primary VQ/LUT token selector, index only.

Faithful to ``true_selfindex_Cache`` in
``baselines/selfindexingkv/models/cache_utils_llama3_selfindex.py`` (the path the
repo's eval scripts actually run — the questCache max/min chunk-bound branch is
dead code no script imports). It is a sign-orthant PRODUCT QUANTIZATION with a
LUT asymmetric-distance selector:

* ``vq_hash`` (cache_utils_llama3_selfindex.py:124-142): the head_dim is split
  into ``SUB=32`` contiguous subspaces of ``SDIM=4`` dims. Each key subvector is
  coded by its 4-bit SIGN pattern (``code in [0,16)``). The codebook entry for
  ``(subspace, code)`` is the MEAN of the key subvectors falling in that sign
  orthant.
* decode (cache_utils_llama3_selfindex.py:322-328): ``table = codebook @ q`` gives
  a ``(SUB, 16)`` per-subspace LUT; each token's approximate ``q.k`` score is
  ``sum_s table[s, code(token, s)]`` (``vq_lutgemv``); the top ``budget`` tokens
  are selected. The GQA group's queries are mean-pooled, so selection is per
  kv-head.

The int32 bit-packed ``vq_lutgemv`` kernel and the small full-precision "full"
buffer (top-64 kept un-quantized) are excluded efficiency/precision machinery;
the selected token SET produced by the LUT approximate-``q.k`` score is what is
reproduced here, fully vectorized. Knob: ``chunk_size`` is no longer used.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..ops import middle_region
from ..registry import IndexMethod, register
from ..types import LayerSelection, MethodConfig, QueryNeeds

SDIM = 4       # dims per subspace -> 4-bit sign code (upstream vq_hash design)
CODES = 16     # 2 ** SDIM sign orthants per subspace
# subspaces = head_dim // SDIM (32 at the real Llama-3.1 head_dim of 128)


@dataclass
class VQIndex:
    codes: torch.Tensor       # (H_kv, N_mid, SUB) long, per-token subspace sign codes
    codebook: torch.Tensor    # (H_kv, SUB, CODES, SDIM) float, orthant-mean codewords
    lo: int


@register(
    "selfindexing",
    kind="per_head",
    needs=QueryNeeds(query="last"),
    reselect="per_step",
    reference="selfindexingkv true_selfindex_Cache: sign-orthant product-quantization LUT top-k per kv-head",
)
class SelfIndexing(IndexMethod):
    def build(self, K, V, cfg: MethodConfig, Q=None):
        H, N, D = K.shape
        if D % SDIM != 0:
            raise ValueError(
                f"selfindexing needs head_dim divisible by {SDIM}, got {D}"
            )
        SUB = D // SDIM                                        # 32 at head_dim 128
        lo, hi = middle_region(N, cfg.sink, cfg.recent)
        Kmid = K[:, lo:hi, :].float()                          # (H, N_mid, D)
        N_mid = Kmid.shape[1]
        if N_mid == 0:
            return VQIndex(
                codes=torch.zeros(H, 0, SUB, dtype=torch.long, device=K.device),
                codebook=torch.zeros(H, SUB, CODES, SDIM, device=K.device),
                lo=lo,
            )
        sub = Kmid.reshape(H, N_mid, SUB, SDIM)                # split head_dim
        # 4-bit sign-orthant code per subspace: bit d = (x_d >= 0)
        powers = (2 ** torch.arange(SDIM, device=K.device)).view(1, 1, 1, SDIM)
        codes = ((sub >= 0).long() * powers).sum(dim=-1)       # (H, N_mid, SUB) in [0,16)
        # codebook[h, s, c, :] = mean of subvectors in subspace s with sign code c
        subvec = sub.permute(0, 2, 1, 3).contiguous()          # (H, SUB, N_mid, SDIM)
        code_hs = codes.permute(0, 2, 1).contiguous()          # (H, SUB, N_mid)
        codebook = torch.zeros(H, SUB, CODES, SDIM, device=K.device, dtype=torch.float32)
        counts = torch.zeros(H, SUB, CODES, 1, device=K.device, dtype=torch.float32)
        codebook.scatter_add_(2, code_hs.unsqueeze(-1).expand(-1, -1, -1, SDIM), subvec)
        counts.scatter_add_(2, code_hs.unsqueeze(-1), torch.ones_like(subvec[..., :1]))
        codebook = codebook / counts.clamp_min(1.0)
        return VQIndex(codes=codes, codebook=codebook, lo=lo)

    def select(self, index: VQIndex, Q, cfg: MethodConfig) -> LayerSelection:
        H, N_mid, _ = index.codes.shape
        g = cfg.group_size
        if N_mid == 0:
            return LayerSelection(
                kind="per_head",
                per_head_idx=torch.empty(H, 0, dtype=torch.long, device=Q.device),
            )
        SUB = index.codebook.shape[1]
        # GQA-group-mean query -> one query per kv-head
        q = Q[:, -1, :].reshape(H, g, -1).mean(dim=1)          # (H, D)
        qsub = q.reshape(H, SUB, SDIM).float()                 # (H, SUB, SDIM)
        # per-subspace LUT: table[h, s, c] = <codeword[h,s,c], q_sub[h,s]>
        table = torch.einsum("hscd,hsd->hsc", index.codebook, qsub)   # (H, SUB, CODES)
        # per-token approximate q.k = sum_s table[s, code(token, s)]
        code_hs = index.codes.permute(0, 2, 1)                 # (H, SUB, N_mid)
        gathered = torch.gather(table, 2, code_hs)             # (H, SUB, N_mid)
        scores = gathered.sum(dim=1)                           # (H, N_mid)
        kb = min(cfg.budget, N_mid)
        idx = scores.topk(kb, dim=-1).indices + index.lo       # (H, kb)
        return LayerSelection(kind="per_head", per_head_idx=idx)
