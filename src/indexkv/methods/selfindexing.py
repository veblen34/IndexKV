"""Self-Indexing KVCache — the repo's primary VQ/LUT token selector, index only.

Faithful to ``true_selfindex_Cache`` in
``baselines/selfindexingkv/models/cache_utils_llama3_selfindex.py`` (the path the
repo's eval scripts actually run — the questCache max/min chunk-bound branch is
dead code no script imports). It is a sign-orthant PRODUCT QUANTIZATION with a
LUT asymmetric-distance selector:

* Before ``vq_hash``, upstream subtracts the per-channel prompt median from K.
  This changes sign-orthant assignments while preserving exact dot-product
  rankings (the removed query-dependent term is constant across tokens).
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
buffer (top-64 kept un-quantized) remain outside the common IndexKV selection
contract.  Exact KV attention remains the default.  For source-fidelity quality
diagnostics, ``emulate_2bit_kv=true`` applies the upstream blockwise 2-bit
key-magnitude/value quantize-dequantize rule after gather.  It does not pack the
cache or reproduce fused-kernel speed/memory.  Knob: ``chunk_size`` is no longer
used.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..ops import middle_region
from ..registry import IndexMethod, register
from ..types import LayerSelection, MethodConfig, QueryNeeds

SDIM = 4       # dims per subspace -> 4-bit sign code (upstream vq_hash design)
CODES = 16     # 2 ** SDIM sign orthants per subspace
QUANT_BLOCK = 32
# subspaces = head_dim // SDIM (32 at the real Llama-3.1 head_dim of 128)


@dataclass
class VQIndex:
    codes: torch.Tensor       # (H_kv, N_mid, SUB) long, per-token subspace sign codes
    codebook: torch.Tensor    # (H_kv, SUB, CODES, SDIM) float, orthant-mean codewords
    lo: int
    hi: int
    key_median: torch.Tensor  # (H_kv, 1, D), zero when centering is disabled
    key_absmax: torch.Tensor  # (H_kv, 1, D), upstream key normalization
    emulate_2bit_kv: bool = False


def _two_bit_minmax(x: torch.Tensor, *, magnitude: bool) -> torch.Tensor:
    """Numerically emulate upstream per-token, per-32-channel 2-bit Q/DQ.

    Keys quantize magnitudes and store sign separately; values use signed
    min/max quantization.  This intentionally returns a dense tensor and is a
    quality model, not a packed-cache implementation.
    """
    if x.shape[-1] % QUANT_BLOCK:
        raise ValueError(
            f"2-bit KV emulation needs head_dim divisible by {QUANT_BLOCK}, "
            f"got {x.shape[-1]}"
        )
    original_dtype = x.dtype
    blocks = x.float().reshape(*x.shape[:-1], -1, QUANT_BLOCK)
    if magnitude:
        signs = torch.where(blocks > 0, 1.0, -1.0)
        values = blocks.abs()
    else:
        signs = None
        values = blocks
    minimum = values.amin(dim=-1, keepdim=True)
    maximum = values.amax(dim=-1, keepdim=True)
    scale = (maximum - minimum) / 3.0
    nonconstant = scale > 0
    safe_scale = torch.where(nonconstant, scale, torch.ones_like(scale))
    levels = torch.round((values - minimum) / safe_scale).clamp_(0, 3)
    # Upstream chooses integer levels with float intermediates, then stores the
    # scale/zero-point in the cache dtype before dequantization.
    stored_scale = safe_scale.to(original_dtype).float()
    stored_minimum = minimum.to(original_dtype).float()
    dequant = levels * stored_scale + stored_minimum
    dequant = torch.where(nonconstant, dequant, values)
    if signs is not None:
        dequant = dequant * signs
    return dequant.reshape_as(x).to(original_dtype)


def _per_output_head(tensor: torch.Tensor, output_heads: int) -> torch.Tensor:
    """Broadcast an H_kv statistic to either H_kv or H_q gathered rows."""
    kv_heads = tensor.shape[0]
    if output_heads == kv_heads:
        return tensor
    if output_heads % kv_heads:
        raise ValueError(
            f"gathered KV heads={output_heads} is not divisible by index "
            f"KV heads={kv_heads}"
        )
    return tensor.repeat_interleave(output_heads // kv_heads, dim=0)


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
        key_centering = cfg.get("key_centering", "median")
        if key_centering not in ("median", "none"):
            raise ValueError(
                "selfindexing.key_centering must be 'median' or 'none', got "
                f"{key_centering!r}"
            )
        emulate_2bit_kv = cfg.get("emulate_2bit_kv", False)
        if not isinstance(emulate_2bit_kv, bool):
            raise ValueError(
                "selfindexing.emulate_2bit_kv must be true or false, got "
                f"{emulate_2bit_kv!r}"
            )
        if emulate_2bit_kv and D % QUANT_BLOCK:
            raise ValueError(
                f"2-bit KV emulation needs head_dim divisible by {QUANT_BLOCK}, got {D}"
            )

        SUB = D // SDIM                                        # 32 at head_dim 128
        key_median = (
            K.median(dim=1, keepdim=True).values
            if key_centering == "median"
            else torch.zeros(H, 1, D, dtype=K.dtype, device=K.device)
        )
        Kcentered = K - key_median
        lo, hi = middle_region(N, cfg.sink, cfg.recent)
        Kmid_source = Kcentered[:, lo:hi, :]
        Kmid = Kmid_source.float()                             # (H, N_mid, D)
        N_mid = Kmid.shape[1]
        key_absmax = (
            Kmid_source.abs().amax(dim=1, keepdim=True)
            if N_mid
            else torch.ones(H, 1, D, dtype=K.dtype, device=K.device)
        )
        if N_mid == 0:
            return VQIndex(
                codes=torch.zeros(H, 0, SUB, dtype=torch.long, device=K.device),
                codebook=torch.zeros(H, SUB, CODES, SDIM, device=K.device),
                lo=lo,
                hi=hi,
                key_median=key_median,
                key_absmax=key_absmax,
                emulate_2bit_kv=emulate_2bit_kv,
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
        return VQIndex(
            codes=codes,
            codebook=codebook,
            lo=lo,
            hi=hi,
            key_median=key_median,
            key_absmax=key_absmax,
            emulate_2bit_kv=emulate_2bit_kv,
        )

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

    def has_kv_transform(self, index: VQIndex, cfg: MethodConfig) -> bool:
        return index.emulate_2bit_kv

    def transform_selected_kv(
        self,
        index: VQIndex,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        cfg: MethodConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not index.emulate_2bit_kv:
            return k, v
        if k.shape != v.shape or k.ndim != 4 or k.shape[0] != 1:
            raise ValueError(
                "selfindexing KV emulation expects matching (1,H,T,D) K/V, got "
                f"K={tuple(k.shape)} V={tuple(v.shape)}"
            )
        output_heads = k.shape[1]
        median = _per_output_head(index.key_median, output_heads).unsqueeze(0)
        absmax = _per_output_head(index.key_absmax, output_heads).unsqueeze(0)
        centered = k - median.to(device=k.device, dtype=k.dtype)
        absmax = absmax.to(device=k.device, dtype=k.dtype)
        safe_absmax = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
        quantized_k = _two_bit_minmax(
            centered / safe_absmax, magnitude=True
        ) * absmax
        quantized_v = _two_bit_minmax(v, magnitude=False)

        if positions.ndim == 1:
            exact = ((positions < index.lo) | (
                (positions >= index.hi) & (positions < cfg.n_prompt)
            )).view(1, 1, -1, 1)
        elif positions.ndim == 2 and positions.shape[0] == output_heads:
            exact = ((positions < index.lo) | (
                (positions >= index.hi) & (positions < cfg.n_prompt)
            )).view(1, output_heads, -1, 1)
        else:
            raise ValueError(
                "gather positions must have shape (T,) or (H,T), got "
                f"{tuple(positions.shape)}"
            )
        # IndexKV's common sink/recent windows remain exact.  All keys are
        # centered so the shared softmax-invariant offset is consistent across
        # exact, quantized and generated positions.
        k_out = torch.where(exact, centered, quantized_k)
        v_out = torch.where(exact, v, quantized_v)
        return k_out, v_out
