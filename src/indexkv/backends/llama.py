"""Shared Llama execution for the index-selection comparisons.

Q/K/V capture and per-step sparse decode are isolated behind the
:class:`ModelBackend` interface. Nothing outside this file imports
``transformers.models.llama``, so every comparison method follows the same
model path.

Handles both attention-forward conventions: the pre-4.48 3-tuple return with a
``past_key_value`` kwarg (e.g. 4.46) and the newer 2-tuple return with
``past_key_values`` — detected once from the installed signature.
"""

from __future__ import annotations

import contextlib
import gc
import inspect
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from .base import Capture, ModelBackend, ModelDims, SelectionProvider


class _QKVCapture:
    """Capture post-RoPE Q/K (+ pre-RoPE V via a projection hook) per layer.

    Attention hooks identify the real layer owning each RoPE call, and counters
    enforce exactly one Q/K capture per layer. V has no RoPE, so a projection
    hook captures it only when a method needs values.
    """

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int,
                 to_cpu: bool = True, need_value: bool = False,
                 q_window: Optional[int] = None):
        if q_window is not None and q_window < 0:
            raise ValueError(f"q_window must be None or >= 0, got {q_window}")
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.to_cpu = to_cpu
        self.need_value = need_value
        self.layers: dict[int, dict[str, torch.Tensor]] = {}
        self._count = 0
        self._v_count = 0
        self._qk_counts: dict[int, int] = {}
        self._v_counts: dict[int, int] = {}
        self._active_layer: Optional[int] = None
        self._attn_counts: dict[int, int] = {}
        # how many trailing prompt positions of Q to KEEP. None -> all (legacy);
        # 0 -> none (per-step-only sweeps never read capture.Q); W -> last W (the
        # max tail a static method's QueryNeeds slice can ask for, e.g. chunkkv's
        # obs_window). K/V are always kept in full — index build needs every key.
        self.q_window = q_window

    def _mv(self, t: torch.Tensor) -> torch.Tensor:
        # A staged CPU tensor must own its storage even when capture runs on CPU:
        # Tensor.cpu() would otherwise be a no-op and a Q slice could pin the
        # full-length allocation. On-device, contiguous() copies a non-contiguous
        # trailing Q slice while avoiding an extra copy for an already-full K.
        t = t.detach()
        if self.to_cpu:
            return t.to(device="cpu", copy=True).contiguous()
        return t.contiguous()

    def _empty_q(self, q_post: torch.Tensor) -> torch.Tensor:
        """Return shape-compatible Q with a genuinely zero-byte allocation.

        An empty slice still retains the complete prefill-Q storage, and
        contiguous() may return that view unchanged. A fresh empty tensor makes
        q_window=0 release the full allocation.
        """
        shape = (q_post.shape[0], q_post.shape[1], 0, q_post.shape[3])
        device = torch.device("cpu") if self.to_cpu else q_post.device
        return torch.empty(shape, dtype=q_post.dtype, device=device)

    def _retain_q(self, q_post: torch.Tensor) -> torch.Tensor:
        if self.q_window is None:
            return self._mv(q_post)
        if self.q_window == 0:
            return self._empty_q(q_post)
        return self._mv(q_post[:, :, -self.q_window:, :])

    def validate(self) -> None:
        """Fail if a prefill did not capture every real layer exactly once."""
        expected = set(range(self.n_layers))
        actual = set(self.layers)
        problems = []
        if self._active_layer is not None:
            problems.append(f"layer {self._active_layer} did not finish attention forward")
        if self._count != self.n_layers:
            problems.append(f"RoPE calls={self._count}, expected={self.n_layers}")
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            problems.append(f"layer ids missing={missing}, extra={extra}")
        for layer in sorted(expected):
            attn_count = self._attn_counts.get(layer, 0)
            if attn_count != 1:
                problems.append(f"layer {layer} attention forwards={attn_count}, expected=1")
            qk_count = self._qk_counts.get(layer, 0)
            if qk_count != 1:
                problems.append(f"layer {layer} Q/K captures={qk_count}, expected=1")
            fields = self.layers.get(layer, {})
            missing_fields = {"q_post", "k_post"} - set(fields)
            if missing_fields:
                problems.append(f"layer {layer} missing {sorted(missing_fields)}")
            if self.need_value:
                v_count = self._v_counts.get(layer, 0)
                if v_count != 1:
                    problems.append(f"layer {layer} V captures={v_count}, expected=1")
                if "v" not in fields:
                    problems.append(f"layer {layer} missing ['v']")
        if problems:
            raise RuntimeError("invalid Llama prefill capture: " + "; ".join(problems))

    @contextlib.contextmanager
    def attach(self, model):
        orig_rope = modeling_llama.apply_rotary_pos_emb
        self.layers.clear()
        self._count = 0
        self._v_count = 0
        self._attn_counts.clear()
        self._qk_counts.clear()
        self._v_counts.clear()
        self._active_layer = None

        model_layers = list(model.model.layers)
        if len(model_layers) != self.n_layers:
            raise RuntimeError(
                "Llama capture/model layer mismatch: "
                f"backend expects {self.n_layers}, model exposes {len(model_layers)}"
            )
        for li, layer in enumerate(model_layers):
            if not hasattr(layer, "self_attn"):
                raise RuntimeError(f"Llama model layer {li} has no self_attn module")
            if self.need_value and not hasattr(layer.self_attn, "v_proj"):
                raise RuntimeError(
                    f"Llama model layer {li} self_attn has no v_proj module"
                )

        def make_attn_pre_hook(layer_idx):
            def hook(mod, args):
                if self._active_layer is not None:
                    raise RuntimeError(
                        "nested Llama attention during capture: "
                        f"entered layer {layer_idx} while layer "
                        f"{self._active_layer} is active"
                    )
                seen = self._attn_counts.get(layer_idx, 0)
                if seen:
                    raise RuntimeError(
                        f"Llama layer {layer_idx} ran attention more than once "
                        "during one prefill"
                    )
                self._attn_counts[layer_idx] = seen + 1
                self._active_layer = layer_idx
            return hook

        def make_attn_post_hook(layer_idx):
            def hook(mod, args, out):
                if self._active_layer != layer_idx:
                    raise RuntimeError(
                        f"Llama capture layer tracking lost at layer {layer_idx}; "
                        f"active={self._active_layer}"
                    )
                self._active_layer = None
            return hook

        def patched_rope(q, k, cos, sin, *args, **kwargs):
            layer = self._active_layer
            if layer is None:
                raise RuntimeError(
                    "apply_rotary_pos_emb called outside a tracked Llama layer"
                )
            seen = self._qk_counts.get(layer, 0)
            if seen:
                raise RuntimeError(
                    f"Llama layer {layer} applied RoPE more than once "
                    "during one prefill"
                )
            q_post, k_post = orig_rope(q, k, cos, sin, *args, **kwargs)
            if q_post.ndim != 4 or k_post.ndim != 4:
                raise RuntimeError(
                    f"Llama layer {layer} returned non-4D Q/K: "
                    f"Q={tuple(q_post.shape)}, K={tuple(k_post.shape)}"
                )
            if q_post.shape[0] != 1 or k_post.shape[0] != 1:
                raise RuntimeError(
                    "Llama capture supports batch size 1, got "
                    f"Q batch={q_post.shape[0]}, K batch={k_post.shape[0]}"
                )
            if (
                k_post.shape[1] != self.n_kv_heads
                or k_post.shape[-1] != self.head_dim
            ):
                raise RuntimeError(
                    f"Llama layer {layer} K shape {tuple(k_post.shape)} "
                    f"does not match H_kv={self.n_kv_heads}, D={self.head_dim}"
                )
            if (
                q_post.shape[2] != k_post.shape[2]
                or q_post.shape[-1] != self.head_dim
            ):
                raise RuntimeError(
                    f"Llama layer {layer} incompatible Q/K shapes: "
                    f"Q={tuple(q_post.shape)}, K={tuple(k_post.shape)}"
                )
            self._qk_counts[layer] = seen + 1
            self._count += 1
            self.layers.setdefault(layer, {})
            self.layers[layer]["q_post"] = self._retain_q(q_post)
            self.layers[layer]["k_post"] = self._mv(k_post)
            return q_post, k_post

        def make_v_hook(layer_idx):
            def hook(mod, inp, out):
                if self._active_layer != layer_idx:
                    raise RuntimeError(
                        f"Llama layer {layer_idx} projected V while "
                        f"active layer is {self._active_layer}"
                    )
                seen = self._v_counts.get(layer_idx, 0)
                if seen:
                    raise RuntimeError(
                        f"Llama layer {layer_idx} projected V more than once "
                        "during one prefill"
                    )
                if out.ndim != 3 or out.shape[0] != 1:
                    raise RuntimeError(
                        "Llama capture supports 3D batch-one V projections, "
                        f"got layer {layer_idx} shape {tuple(out.shape)}"
                    )
                expected_width = self.n_kv_heads * self.head_dim
                if out.shape[-1] != expected_width:
                    raise RuntimeError(
                        f"Llama layer {layer_idx} V width={out.shape[-1]}, "
                        f"expected={expected_width}"
                    )
                bsz, q_len, _ = out.shape
                v = out.view(
                    bsz, q_len, self.n_kv_heads, self.head_dim
                ).transpose(1, 2)
                self._v_counts[layer_idx] = seen + 1
                self._v_count += 1
                self.layers.setdefault(layer_idx, {})
                self.layers[layer_idx]["v"] = self._mv(v)
            return hook

        attn_handles = []
        v_handles = []
        for li, layer in enumerate(model_layers):
            attn_handles.append(
                layer.self_attn.register_forward_pre_hook(make_attn_pre_hook(li))
            )
            attn_handles.append(
                layer.self_attn.register_forward_hook(make_attn_post_hook(li))
            )
            if self.need_value:
                v_handles.append(
                    layer.self_attn.v_proj.register_forward_hook(make_v_hook(li))
                )

        modeling_llama.apply_rotary_pos_emb = patched_rope
        try:
            yield self
            self.validate()
        finally:
            modeling_llama.apply_rotary_pos_emb = orig_rope
            for handle in attn_handles:
                handle.remove()
            for handle in v_handles:
                handle.remove()


class _ChunkedPrefillMLP:
    """Bound prefill MLP workspace without changing token-wise semantics.

    Llama's MLP is independent across the sequence dimension. Evaluating that
    dimension in chunks avoids materializing full-length gate/up projections
    while preserving the same weights, dtype, token order, and attention path.
    The patch is scoped to a context manager so single-token decode always uses
    the model's original forward methods.
    """

    def __init__(self, chunk_size: Optional[int]):
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(
                f"prefill MLP chunk size must be > 0 or None, got {chunk_size}"
            )
        self.chunk_size = chunk_size

    def _make_forward(self, original):
        chunk_size = self.chunk_size

        def forward(hidden_states, *args, **kwargs):
            if chunk_size is None or hidden_states.shape[-2] <= chunk_size:
                return original(hidden_states, *args, **kwargs)
            outputs = [
                original(chunk, *args, **kwargs)
                for chunk in hidden_states.split(chunk_size, dim=-2)
            ]
            return torch.cat(outputs, dim=-2)

        return forward

    @contextlib.contextmanager
    def attach(self, model):
        if self.chunk_size is None:
            yield
            return

        originals = {}
        try:
            for layer_idx, layer in enumerate(model.model.layers):
                originals[layer_idx] = layer.mlp.forward
                layer.mlp.forward = self._make_forward(layer.mlp.forward)
            yield
        finally:
            for layer_idx, original in originals.items():
                model.model.layers[layer_idx].mlp.forward = original


def _validate_gather_inputs(q, k, v, allow) -> torch.Tensor:
    """Validate decode-time GQA tensors and return allow as (Hm, T)."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"q/k/v must be 4D, got {q.ndim}D/{k.ndim}D/{v.ndim}D"
        )
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise ValueError(
            "direct sparse gather supports batch size 1, got "
            f"q/k/v batches={q.shape[0]}/{k.shape[0]}/{v.shape[0]}"
        )
    if k.shape != v.shape:
        raise ValueError(f"K/V shapes differ: K={tuple(k.shape)}, V={tuple(v.shape)}")
    _, H_q, q_len, D = q.shape
    _, H_kv, T_k, kv_dim = k.shape
    if q_len != 1:
        raise ValueError(f"direct sparse gather requires q_len=1, got {q_len}")
    if D != kv_dim:
        raise ValueError(f"Q/K head dims differ: Q={D}, K={kv_dim}")
    if H_kv <= 0 or H_q % H_kv:
        raise ValueError(f"H_q={H_q} must be divisible by H_kv={H_kv}")
    if q.device != k.device or q.device != v.device:
        raise ValueError(
            f"q/k/v devices differ: {q.device}/{k.device}/{v.device}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"q/k/v dtypes differ: {q.dtype}/{k.dtype}/{v.dtype}")
    if allow.dtype != torch.bool:
        raise TypeError(f"allow must be bool, got {allow.dtype}")
    if allow.device != q.device:
        raise ValueError(f"allow is on {allow.device}, q/k/v are on {q.device}")
    if (
        allow.ndim != 4
        or allow.shape[0] != 1
        or allow.shape[2] != 1
        or allow.shape[3] != T_k
        or allow.shape[1] not in (1, H_q)
    ):
        raise ValueError(
            "allow must have shape (1, 1|H_q, 1, T_k), got "
            f"{tuple(allow.shape)} for H_q={H_q}, T_k={T_k}"
        )
    a = allow[0, :, 0, :]
    nonempty = a.any(dim=-1)
    if not bool(nonempty.all()):
        rows = torch.where(~nonempty)[0].tolist()
        raise ValueError(
            "each query-head selection must retain at least one token; "
            f"empty mask rows={rows}. The benchmark sink/recent contract "
            "should make zero-selection impossible."
        )
    return a


def _sdpa_gqa(q, k, v, attn_mask=None):
    """Call SDPA without materializing repeated GQA keys/values."""
    if q.shape[1] == k.shape[1]:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False
        )
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=False, enable_gqa=True
    )


@torch.no_grad()
def _gather_sdpa(q, k, v, allow, kv_transform=None):
    """Sparse decode directly from the unrepeated (1, H_kv, T_k, D) cache.

    Shared masks gather each selected token once per KV head and use native GQA.
    Per-query-head masks map query head ``h`` to KV head
    ``h // (H_q / H_kv)`` and gather only a compact, possibly ragged,
    ``(1, H_q, M, D)`` tensor. No full-width ``repeat_kv`` is materialized.

    A zero-selection row is rejected: attention over an empty key set has no
    defined softmax semantics and violates the benchmark's common-window policy.
    """
    a = _validate_gather_inputs(q, k, v, allow)
    _, H_q, _, D = q.shape
    _, H_kv, T_k, _ = k.shape

    # A per-query-head mask may still be identical across all heads. Treat it as
    # shared so GQA keys are gathered once rather than once per query head.
    if a.shape[0] == H_q and torch.equal(a, a[:1].expand_as(a)):
        a = a[:1]

    if a.shape[0] == 1:
        count = int(a.sum().item())
        if count == T_k:
            idx = torch.arange(T_k, device=q.device)
            k_sel, v_sel = k, v
        else:
            idx = torch.nonzero(a[0], as_tuple=False).flatten()
            gidx = idx.view(1, 1, count, 1).expand(1, H_kv, count, D)
            k_sel = torch.gather(k, 2, gidx)
            v_sel = torch.gather(v, 2, gidx)
        if kv_transform is not None:
            k_sel, v_sel = kv_transform(k_sel, v_sel, idx)
        return _sdpa_gqa(q, k_sel, v_sel)

    counts = a.sum(dim=-1)
    M = int(counts.max().item())
    # topk on 0/1 rows puts selected positions before ragged padding.
    vals, idx = torch.topk(a.to(torch.int8), M, dim=-1)
    keep = vals.bool()
    groups = H_q // H_kv
    kv_head = torch.arange(H_q, device=q.device) // groups
    # Advanced indexing reads exactly H_q*M rows from the H_kv cache; it does not
    # construct an H_q*T_k repeat before selection.
    k_sel = k[0, kv_head[:, None], idx].unsqueeze(0)
    v_sel = v[0, kv_head[:, None], idx].unsqueeze(0)
    if kv_transform is not None:
        k_sel, v_sel = kv_transform(k_sel, v_sel, idx)
    add = None
    if not bool(keep.all()):
        add = torch.zeros(1, H_q, 1, M, dtype=q.dtype, device=q.device)
        add = add.masked_fill(~keep.view(1, H_q, 1, M), float("-inf"))
    return F.scaled_dot_product_attention(
        q, k_sel, v_sel, attn_mask=add, is_causal=False
    )


class _SparseAttn:
    """Patch ``LlamaAttention.forward`` during decode to honor per-layer masks.

    Prefill is left on the model's default sdpa. During decode the query length
    is 1. With ``fast=True`` (default) the selected KV is gathered so attention
    only touches ~budget keys instead of all T_k; with ``fast=False`` the legacy
    full-width ``sdpa`` + additive -inf mask path runs (kept for equivalence
    checks). At every decode step the provider is asked for this layer's mask
    with the live post-RoPE query — static methods return their frozen mask,
    per-step methods re-select. Masks cover the prompt; generated positions are
    always allowed.
    """

    def __init__(self, provider: Optional[SelectionProvider] = None, fast: bool = True):
        self.provider = provider
        self.fast = fast

    @contextlib.contextmanager
    def attach(self, model):
        orig = {}
        for layer_idx, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            orig[layer_idx] = attn.forward
            attn.forward = self._make_forward(attn, layer_idx)
        try:
            yield self
        finally:
            for layer_idx, layer in enumerate(model.model.layers):
                model.model.layers[layer_idx].self_attn.forward = orig[layer_idx]

    def _make_forward(self, attn_module, layer_idx: int):
        ctx = self
        cfg = attn_module.config
        H_q = cfg.num_attention_heads
        H_kv = cfg.num_key_value_heads
        head_dim = getattr(attn_module, "head_dim", None) or cfg.head_dim
        kv_groups = H_q // H_kv
        # pre-4.48 attention takes `past_key_value` and must return a 3-tuple
        legacy = "past_key_value" in inspect.signature(
            modeling_llama.LlamaAttention.forward).parameters

        def forward(hidden_states, position_embeddings=None, attention_mask=None,
                    past_key_values=None, past_key_value=None,
                    cache_position=None, **kwargs):
            if past_key_values is None:
                past_key_values = past_key_value
            bsz, q_len, _ = hidden_states.size()
            if bsz != 1:
                raise ValueError(f"sparse Llama attention requires batch size 1, got {bsz}")
            q = attn_module.q_proj(hidden_states).view(bsz, q_len, H_q, head_dim).transpose(1, 2)
            k = attn_module.k_proj(hidden_states).view(bsz, q_len, H_kv, head_dim).transpose(1, 2)
            v = attn_module.v_proj(hidden_states).view(bsz, q_len, H_kv, head_dim).transpose(1, 2)

            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                k, v = past_key_values.update(k, v, layer_idx, cache_kwargs)

            T_k = k.size(-2)

            allow = None
            if ctx.provider is not None and q_len == 1:
                allow = ctx.provider.mask_for(layer_idx, q[0])
                if allow is None:
                    # DenseProvider and global dense-prefix layers use the same
                    # direct-GQA fast implementation with an unrestricted mask.
                    allow = torch.ones(
                        1, 1, 1, T_k, dtype=torch.bool, device=q.device
                    )
                else:
                    # mask covers the prompt; generated positions are attendable
                    if allow.shape[-1] < T_k:
                        pad = torch.ones(
                            allow.shape[0],
                            allow.shape[1],
                            allow.shape[2],
                            T_k - allow.shape[-1],
                            dtype=torch.bool,
                            device=allow.device,
                        )
                        allow = torch.cat([allow, pad], dim=-1)
                    elif allow.shape[-1] > T_k:
                        allow = allow[..., :T_k]

            transform_selected_kv = None
            if ctx.provider is not None:
                has_kv_transform = getattr(
                    ctx.provider, "has_kv_transform", lambda _: False
                )
                if has_kv_transform(layer_idx):
                    transform_selected_kv = getattr(
                        ctx.provider, "transform_selected_kv", None
                    )
            if allow is not None and ctx.fast:
                # Directly gather from H_kv cache. At decode q_len==1 the causal
                # mask allows all past positions, so allow is the only constraint.
                out = _gather_sdpa(
                    q,
                    k,
                    v,
                    allow,
                    kv_transform=(
                        None
                        if transform_selected_kv is None
                        else lambda k_sel, v_sel, positions: (
                            transform_selected_kv(
                                layer_idx, k_sel, v_sel, positions
                            )
                        )
                    ),
                )
            else:
                # Preserve the legacy full-width path for dense prefill/decode and
                # explicit fast=False equivalence checks.
                if allow is not None:
                    _validate_gather_inputs(q, k, v, allow)
                    if transform_selected_kv is not None:
                        positions = torch.arange(T_k, device=q.device)
                        k, v = transform_selected_kv(
                            layer_idx, k, v, positions
                        )
                k_full = repeat_kv(k, kv_groups)
                v_full = repeat_kv(v, kv_groups)
                offset = T_k - q_len
                row = torch.arange(q_len, device=q.device).unsqueeze(-1)
                col = torch.arange(T_k, device=q.device).unsqueeze(0)
                disallow = col > row + offset
                attn_mask = disallow.view(1, 1, q_len, T_k).expand(
                    bsz, 1, q_len, T_k
                )
                if allow is not None:
                    attn_mask = attn_mask | (~allow)
                additive = torch.zeros_like(attn_mask, dtype=q.dtype)
                additive = additive.masked_fill(attn_mask, float("-inf"))
                out = F.scaled_dot_product_attention(
                    q,
                    k_full,
                    v_full,
                    attn_mask=additive,
                    is_causal=False,
                )
            out = out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            out = attn_module.o_proj(out)
            if legacy:
                return out, None, past_key_values
            return out, None

        return forward


class LlamaBackend(ModelBackend):
    """HuggingFace Llama backend (transformers 5.x).

    Capture and sparse decode intentionally support one sequence on one device.
    The public entry points enforce both constraints before touching the model.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        prefill_mlp_chunk_size: Optional[int] = None,
    ):
        self.model = model
        self.tok = tokenizer
        self._requested_revision: Optional[str] = None
        if prefill_mlp_chunk_size is not None and prefill_mlp_chunk_size <= 0:
            raise ValueError(
                "prefill_mlp_chunk_size must be > 0 or None, got "
                f"{prefill_mlp_chunk_size}"
            )
        self.prefill_mlp_chunk_size = prefill_mlp_chunk_size
        cfg = model.config
        self.dims = ModelDims(
            n_layers=cfg.num_hidden_layers,
            n_heads=cfg.num_attention_heads,
            n_kv_heads=cfg.num_key_value_heads,
            head_dim=(
                getattr(cfg, "head_dim", None)
                or cfg.hidden_size // cfg.num_attention_heads
            ),
        )
        self._single_model_device()

    def _single_model_device(self) -> torch.device:
        """Return the only model device or reject sharded/offloaded models.

        The patched attention passes live tensors directly to a provider, so a
        split model would require an explicit cross-device selection contract.
        Until that exists, accepting one would produce ambiguous or late errors.
        """
        devices = {param.device for param in self.model.parameters()}
        devices.update(
            buffer.device
            for buffer in self.model.buffers()
            if buffer.numel() > 0
        )
        if not devices:
            raise ValueError("LlamaBackend requires a model with parameters or buffers")
        if len(devices) != 1:
            formatted = ", ".join(sorted(map(str, devices)))
            raise ValueError(
                f"LlamaBackend supports a single model device, found: {formatted}"
            )
        device = next(iter(devices))
        if device.type == "meta":
            raise ValueError("LlamaBackend cannot run with parameters on the meta device")
        return device

    def _validate_prompt_ids(self, prompt_ids: torch.Tensor) -> torch.device:
        if not isinstance(prompt_ids, torch.Tensor):
            raise TypeError("prompt_ids must be a torch.Tensor")
        if prompt_ids.ndim != 2:
            raise ValueError(
                f"prompt_ids must have shape (1, N), got {tuple(prompt_ids.shape)}"
            )
        if prompt_ids.shape[0] != 1:
            raise ValueError(
                f"LlamaBackend supports batch size 1, got {prompt_ids.shape[0]}"
            )
        if prompt_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"prompt_ids must contain integer token ids, got {prompt_ids.dtype}"
            )
        device = self._single_model_device()
        if prompt_ids.device != device:
            raise ValueError(
                f"prompt_ids are on {prompt_ids.device}, but model is on {device}"
            )
        return device

    @classmethod
    def load(
        cls,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "cuda",
        revision: Optional[str] = None,
        prefill_mlp_chunk_size: Optional[int] = None,
    ) -> "LlamaBackend":
        tok = AutoTokenizer.from_pretrained(model_path, revision=revision)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            device_map=device_map,
            revision=revision,
        ).eval()
        backend = cls(
            model,
            tok,
            prefill_mlp_chunk_size=prefill_mlp_chunk_size,
        )
        backend._requested_revision = revision
        return backend

    def provenance(self) -> dict:
        """Return lightweight model identity without hashing weight tensors."""
        cfg = self.model.config
        commit_hash = getattr(cfg, "_commit_hash", None)
        resolved_revision = (
            str(commit_hash)
            if commit_hash is not None and str(commit_hash).strip()
            else None
        )
        config_name_or_path = getattr(cfg, "_name_or_path", None)
        if config_name_or_path is None:
            config_name_or_path = getattr(cfg, "name_or_path", None)
        tokenizer_name_or_path = getattr(self.tok, "name_or_path", None)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is None:
            first_parameter = next(self.model.parameters(), None)
            model_dtype = (
                first_parameter.dtype if first_parameter is not None else None
            )
        return {
            "config_name_or_path": (
                str(config_name_or_path)
                if config_name_or_path is not None
                else None
            ),
            "config_commit_hash": resolved_revision,
            "tokenizer_name_or_path": (
                str(tokenizer_name_or_path)
                if tokenizer_name_or_path is not None
                else None
            ),
            "requested_revision": self._requested_revision,
            # Never substitute requested_revision here: a tag/branch or local
            # path is not a resolved immutable revision without _commit_hash.
            "resolved_revision": resolved_revision,
            "model_dtype": str(model_dtype) if model_dtype is not None else None,
            "model_device": str(self._single_model_device()),
            "prefill_mlp_chunk_size": self.prefill_mlp_chunk_size,
            "dims": {
                "n_layers": self.dims.n_layers,
                "n_heads": self.dims.n_heads,
                "n_kv_heads": self.dims.n_kv_heads,
                "head_dim": self.dims.head_dim,
            },
        }

    @torch.no_grad()
    def tokenize(self, prompt: str, *, chat_template: bool = False) -> torch.Tensor:
        dev = self._single_model_device()
        if chat_template:
            ids = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if hasattr(ids, "input_ids"):     # transformers >=5.x returns a BatchEncoding
                ids = ids.input_ids
        else:
            ids = self.tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        ids = ids.to(dev)
        self._validate_prompt_ids(ids)
        return ids

    @torch.no_grad()
    def capture(self, prompt_ids: torch.Tensor, *, need_value: bool = False,
                dtype: torch.dtype = torch.bfloat16,
                resident: str = "auto",
                q_window: Optional[int] = None) -> Capture:
        """Prefill once and return per-layer post-RoPE Q/K(/V).

        This is a method-agnostic shared step: the same capture feeds every
        method's ``build``, and its cost is not attributed to any method in the
        sweep metrics. ``resident`` only changes *where* the tensors are staged
        while being collected, never their values:

        * ``"gpu"``  — keep captured Q/K/V on the GPU, skipping the
          GPU->CPU->GPU round trip. Fastest, but raises the prefill peak memory
          (all layers' Q/K live on the GPU at once), so very long contexts OOM.
        * ``"cpu"``  — stage each layer to CPU during prefill, move back after
          (the original behavior). Lower peak memory, slower.
        * ``"auto"`` — try ``"gpu"``; on OOM, free and retry ``"cpu"``. Default.

        ``q_window`` trims how many trailing prompt positions of Q are retained
        (None = all; 0 = none; W = last W). Only STATIC methods read capture.Q,
        and only as a trailing slice (``last`` -> 1, ``obs_window`` -> that many),
        so a sweep passes the max such tail across its runnable methods — 0 when
        all methods are per-step. Q is by far the biggest captured tensor at long
        context (all-layer Q is ~31GB at 120k), so trimming it is what makes the
        full medium/long benchmark fit. K/V are always full (build needs them).

        The output tensors are identical regardless of ``resident``; ``q_window``
        only changes how much of Q is retained, and every method still receives
        exactly the slice its ``QueryNeeds`` declares, so comparability holds.
        """
        self._validate_prompt_ids(prompt_ids)
        if resident == "auto":
            try:
                return self._run_capture(prompt_ids, need_value, dtype, to_cpu=False, q_window=q_window)
            except torch.cuda.OutOfMemoryError:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            return self._run_capture(prompt_ids, need_value, dtype, to_cpu=True, q_window=q_window)
        if resident not in ("gpu", "cpu"):
            raise ValueError(f"resident must be 'auto'|'gpu'|'cpu', got {resident!r}")
        return self._run_capture(prompt_ids, need_value, dtype,
                                 to_cpu=(resident == "cpu"), q_window=q_window)

    @torch.no_grad()
    def _run_capture(self, prompt_ids: torch.Tensor, need_value: bool,
                     dtype: torch.dtype, to_cpu: bool,
                     q_window: Optional[int] = None) -> Capture:
        d = self.dims
        cap = _QKVCapture(d.n_layers, d.n_kv_heads, d.head_dim,
                          to_cpu=to_cpu, need_value=need_value, q_window=q_window)
        with cap.attach(self.model), _ChunkedPrefillMLP(
            self.prefill_mlp_chunk_size
        ).attach(self.model):
            # backbone only — capture needs Q/K, not the lm_head logits (which are
            # ~31GB at 120k tokens and pure waste here). RoPE is patched inside the
            # decoder layers, so Q/K are captured exactly as in a full-model pass.
            self.model.model(input_ids=prompt_ids, use_cache=False)

        dev = self._single_model_device()
        K = {L: cap.layers[L]["k_post"][0].to(dtype=dtype, device=dev) for L in cap.layers}
        Q = {L: cap.layers[L]["q_post"][0].to(dtype=dtype, device=dev) for L in cap.layers}
        V = None
        if need_value:
            V = {L: cap.layers[L]["v"][0].to(dtype=dtype, device=dev) for L in cap.layers}
        del cap
        if to_cpu:
            torch.cuda.empty_cache()
        return Capture(K=K, Q=Q, V=V, n_prompt=int(prompt_ids.shape[1]))

    @torch.no_grad()
    def sparse_generate(self, prompt_ids: torch.Tensor, max_new: int,
                        provider: Optional[SelectionProvider], *, fast: bool = True) -> str:
        if max_new <= 0:
            return ""
        device = self._validate_prompt_ids(prompt_ids)
        model, tok = self.model, self.tok
        with _ChunkedPrefillMLP(self.prefill_mlp_chunk_size).attach(model):
            out = model(
                input_ids=prompt_ids,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        if nxt == tok.eos_token_id:
            return ""
        generated = [nxt]

        def _step(next_tok, past_kv):
            inp = torch.tensor([[next_tok]], device=device)
            o = model(input_ids=inp, use_cache=True, past_key_values=past_kv, return_dict=True)
            return int(o.logits[0, -1].argmax()), o.past_key_values

        if provider is None:
            for _ in range(max_new - 1):
                nxt, past = _step(nxt, past)
                if nxt == tok.eos_token_id:
                    break
                generated.append(nxt)
        else:
            with _SparseAttn(provider=provider, fast=fast).attach(self.model):
                for _ in range(max_new - 1):
                    nxt, past = _step(nxt, past)
                    if nxt == tok.eos_token_id:
                        break
                    generated.append(nxt)
        return tok.decode(torch.tensor(generated), skip_special_tokens=True)
