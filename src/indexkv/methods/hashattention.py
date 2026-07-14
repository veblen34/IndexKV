"""HashAttention (Desai et al., ICML'25) — learned-hash semantic sparsity, index only.

Per-query-head learned MLPs (the ``USA`` module: Linear→SiLU→Linear→SiLU→Linear,
128→128→128→32 for the shipped Llama-3.1-8B patch) map post-RoPE K and Q into
a shared Hamming space; codes use upstream torch.sign values {-1, 0, +1}, and
each key is scored by
the signed inner product ``q_code · k_code`` (= bits_matched − bits_mismatched).
Top ``budget`` per query head, independently — HashAttention operates on
repeat_kv'd keys, so unlike hata each query head in a GQA group gets its own
selection (hashattention_llama.py:151-226, compute_mask 454-493).

Keys are hashed once at build (per layer); the tiny query MLP runs at every
decode step. GPT-Fast / FlashDecode integrations are excluded system
acceleration.

Weights: ``cfg.extra["weights"]`` = {"k": [(W, b) per Linear], "q": [...]}
stacked over heads — see weights.load_hashattention_weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..ops import middle_region
from ..registry import IndexMethod, register
from ..types import LayerSelection, MethodConfig, QueryNeeds


@torch.no_grad()
def _mlp(x: torch.Tensor, linears) -> torch.Tensor:
    """Apply per-head stacked Linears with SiLU between: x (H, N, d_in)."""
    for i, (W, b) in enumerate(linears):
        x = torch.einsum("hnd,hod->hno", x, W.float()) + b.float().unsqueeze(1)
        if i < len(linears) - 1:
            x = F.silu(x)
    return x


@dataclass
class HashAttnIndex:
    codes: torch.Tensor    # (H_q, N_mid, bits) torch.sign key codes
    q_mlp: list            # per-Linear (W (H_q, d_out, d_in), b (H_q, d_out))
    lo: int


@register(
    "hashattention",
    kind="per_head",
    needs=QueryNeeds(query="last", weights_key="hashattention"),
    reselect="per_step",
    reference="HashAttention-1.0 (USA MLP k/q hash codes, signed inner product, per-head top-k)",
)
class HashAttention(IndexMethod):
    def build(self, K, V, cfg: MethodConfig, Q=None):
        w = cfg.get("weights")
        if w is None:
            raise ValueError(
                f"hashattention weights are missing for sparse layer {cfg.layer_idx}"
            )
        H_kv, N, D = K.shape
        lo, hi = middle_region(N, cfg.sink, cfg.recent)
        Kr = K[:, lo:hi, :].repeat_interleave(cfg.group_size, dim=0)  # (H_q, N_mid, D)
        emb = _mlp(Kr.float(), w["k"])
        codes = emb.sign().to(torch.bfloat16)
        return HashAttnIndex(codes=codes, q_mlp=w["q"], lo=lo)

    def select(self, index: HashAttnIndex, Q, cfg: MethodConfig) -> LayerSelection:
        H_q, N_mid, bits = index.codes.shape
        q = Q[:, -1, :].unsqueeze(1)                      # (H_q, 1, D)
        emb = _mlp(q.float(), index.q_mlp)
        qcodes = emb.sign().to(index.codes.dtype)
        score = torch.einsum("hnb,hmb->hn", index.codes, qcodes).float()  # (H_q, N_mid)
        kb = min(cfg.budget, N_mid)
        idx = score.topk(kb, dim=-1).indices + index.lo   # (H_q, kb)
        return LayerSelection(kind="per_head", per_head_idx=idx)
