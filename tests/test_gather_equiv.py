"""Pure-Torch equivalence tests for direct GQA sparse gather.

The fast helper receives the unrepeated ``(1, H_kv, T, D)`` cache.  These tests
compare it with the slow/reference path that explicitly repeats KV to H_q and
applies the full-width additive mask.  They run without Transformers or model
weights: ``python tests/test_gather_equiv.py``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


def _install_transformers_stub() -> None:
    try:
        import transformers  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    transformers = types.ModuleType("transformers")
    models = types.ModuleType("transformers.models")
    llama_pkg = types.ModuleType("transformers.models.llama")
    modeling = types.ModuleType("transformers.models.llama.modeling_llama")

    def identity_rope(q, k, cos, sin, *args, **kwargs):
        return q, k

    def repeat_kv(x, groups):
        return x.repeat_interleave(groups, dim=1)

    class LlamaAttention(torch.nn.Module):
        def forward(self, hidden_states, past_key_value=None):  # pragma: no cover
            raise NotImplementedError

    modeling.apply_rotary_pos_emb = identity_rope
    modeling.repeat_kv = repeat_kv
    modeling.LlamaAttention = LlamaAttention
    llama_pkg.modeling_llama = modeling
    models.llama = llama_pkg
    transformers.models = models
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object

    sys.modules["transformers"] = transformers
    sys.modules["transformers.models"] = models
    sys.modules["transformers.models.llama"] = llama_pkg
    sys.modules["transformers.models.llama.modeling_llama"] = modeling


_install_transformers_stub()
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

# Import the backend without executing indexkv/__init__.py. This keeps the test
# independent of method-registration work elsewhere in the repository.
if "indexkv" not in sys.modules:
    indexkv_pkg = types.ModuleType("indexkv")
    indexkv_pkg.__path__ = [str(SRC_ROOT / "indexkv")]
    backends_pkg = types.ModuleType("indexkv.backends")
    backends_pkg.__path__ = [str(SRC_ROOT / "indexkv" / "backends")]
    sys.modules["indexkv"] = indexkv_pkg
    sys.modules["indexkv.backends"] = backends_pkg

from indexkv.backends import llama as llama_backend  # noqa: E402


T, D = 37, 16


def _reference_full_width(q, k, v, allow):
    """Legacy slow path: repeat to H_q, then full-width masked SDPA."""
    H_q = q.shape[1]
    H_kv = k.shape[1]
    groups = H_q // H_kv
    k_full = k.repeat_interleave(groups, dim=1)
    v_full = v.repeat_interleave(groups, dim=1)
    a = allow
    if allow.shape[1] == 1:
        a = allow.expand(1, H_q, 1, allow.shape[-1])
    additive = torch.zeros(
        1, H_q, 1, allow.shape[-1], dtype=q.dtype, device=q.device
    )
    additive = additive.masked_fill(~a, float("-inf"))
    return F.scaled_dot_product_attention(
        q, k_full, v_full, attn_mask=additive, is_causal=False
    )


def _make_allow(kind: str, H_q: int, device: torch.device) -> torch.Tensor:
    if kind == "shared_all":
        return torch.ones(1, 1, 1, T, dtype=torch.bool, device=device)
    if kind == "per_head_all":
        return torch.ones(1, H_q, 1, T, dtype=torch.bool, device=device)

    Hm = 1 if kind == "shared_partial" else H_q
    allow = torch.zeros(Hm, T, dtype=torch.bool)
    if kind == "shared_partial":
        allow[0, torch.tensor([0, 2, 5, 9, 14, 21, 30, 36])] = True
    elif kind == "per_head_uniform":
        for head in range(H_q):
            positions = (torch.arange(9) * 4 + head) % T
            allow[head, positions] = True
    elif kind == "per_head_ragged":
        for head in range(H_q):
            count = 2 + (head * 3) % 11
            positions = (torch.arange(count) * 5 + 2 * head) % T
            allow[head, positions] = True
    else:  # pragma: no cover - test construction guard
        raise ValueError(kind)
    return allow.view(1, Hm, 1, T).to(device)


def _case_tensors(H_q: int, H_kv: int, dtype, device):
    generator = torch.Generator().manual_seed(20260711 + H_q + H_kv)
    q = torch.randn(1, H_q, 1, D, generator=generator).to(device, dtype)
    k = torch.randn(1, H_kv, T, D, generator=generator).to(device, dtype)
    v = torch.randn(1, H_kv, T, D, generator=generator).to(device, dtype)
    return q, k, v


def test_direct_gather_matches_full_width_reference() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtypes = [(torch.float32, 3e-5, 3e-6)]
    if device.type == "cuda":
        dtypes.append((torch.bfloat16, 3e-2, 3e-2))

    # H_q == H_kv is MHA; H_q > H_kv is grouped-query attention.
    head_layouts = [(4, 4, "mha"), (8, 2, "gqa")]
    masks = [
        "shared_all",
        "per_head_all",
        "shared_partial",
        "per_head_uniform",
        "per_head_ragged",
    ]
    for dtype, rtol, atol in dtypes:
        for H_q, H_kv, layout in head_layouts:
            q, k, v = _case_tensors(H_q, H_kv, dtype, device)
            assert k.shape[1] == H_kv  # the fast input is deliberately unrepeated
            for mask_kind in masks:
                allow = _make_allow(mask_kind, H_q, device)
                expected = _reference_full_width(q, k, v, allow)
                actual = llama_backend._gather_sdpa(q, k, v, allow)
                torch.testing.assert_close(
                    actual, expected, rtol=rtol, atol=atol,
                    msg=lambda msg: f"{layout}/{dtype}/{mask_kind}: {msg}",
                )


def test_direct_gather_never_calls_repeat_kv() -> None:
    q, k, v = _case_tensors(8, 2, torch.float32, torch.device("cpu"))
    allow = _make_allow("per_head_ragged", 8, torch.device("cpu"))
    original = llama_backend.repeat_kv

    def forbidden_repeat(*args, **kwargs):  # pragma: no cover - must stay unused
        raise AssertionError("fast gather called repeat_kv")

    llama_backend.repeat_kv = forbidden_repeat
    try:
        llama_backend._gather_sdpa(q, k, v, allow)
    finally:
        llama_backend.repeat_kv = original


def test_direct_gather_exposes_selected_positions_to_kv_transform() -> None:
    q, k, v = _case_tensors(8, 2, torch.float32, torch.device("cpu"))
    allow = _make_allow("shared_partial", 8, torch.device("cpu"))
    expected_positions = torch.nonzero(
        allow[0, 0, 0], as_tuple=False
    ).flatten()
    seen = {}

    def zero_value_transform(k_sel, v_sel, positions):
        seen["positions"] = positions.clone()
        return k_sel, torch.zeros_like(v_sel)

    output = llama_backend._gather_sdpa(
        q, k, v, allow, kv_transform=zero_value_transform
    )
    assert torch.equal(seen["positions"], expected_positions)
    assert torch.equal(output, torch.zeros_like(output))


def test_sparse_forward_fast_avoids_repeat_and_matches_slow() -> None:
    H_q, H_kv, head_dim, width, cache_len = 8, 2, 4, 32, 9

    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(
                num_attention_heads=H_q,
                num_key_value_heads=H_kv,
                head_dim=head_dim,
            )
            self.head_dim = head_dim
            self.q_proj = torch.nn.Linear(width, H_q * head_dim, bias=False)
            self.k_proj = torch.nn.Linear(width, H_kv * head_dim, bias=False)
            self.v_proj = torch.nn.Linear(width, H_kv * head_dim, bias=False)
            self.o_proj = torch.nn.Linear(width, width, bias=False)

    class FixedCache:
        def __init__(self, k, v):
            self.k = k
            self.v = v

        def update(self, k, v, layer_idx, cache_kwargs):
            return self.k, self.v

    class Provider:
        def __init__(self, allow):
            self.allow = allow

        def mask_for(self, layer_idx, q):
            return self.allow

    torch.manual_seed(17)
    attention = FakeAttention()
    hidden = torch.randn(1, 1, width)
    k_cache = torch.randn(1, H_kv, cache_len, head_dim)
    v_cache = torch.randn(1, H_kv, cache_len, head_dim)
    cache = FixedCache(k_cache, v_cache)
    allow = torch.zeros(1, 1, 1, cache_len, dtype=torch.bool)
    allow[..., torch.tensor([0, 2, 5, 8])] = True
    provider = Provider(allow)

    original_rope = llama_backend.apply_rotary_pos_emb
    original_repeat = llama_backend.repeat_kv
    llama_backend.apply_rotary_pos_emb = (
        lambda q, k, cos, sin: (q, k)
    )

    def forbidden_repeat(*args, **kwargs):  # pragma: no cover - must stay unused
        raise AssertionError("fast _SparseAttn.forward called repeat_kv")

    try:
        fast_forward = llama_backend._SparseAttn(
            provider=provider, fast=True
        )._make_forward(attention, layer_idx=0)
        llama_backend.repeat_kv = forbidden_repeat
        fast = fast_forward(
            hidden,
            position_embeddings=(None, None),
            past_key_values=cache,
        )[0]

        llama_backend.repeat_kv = original_repeat
        slow_forward = llama_backend._SparseAttn(
            provider=provider, fast=False
        )._make_forward(attention, layer_idx=0)
        slow = slow_forward(
            hidden,
            position_embeddings=(None, None),
            past_key_values=cache,
        )[0]

        # DenseProvider and dense-prefix providers return None. A non-None
        # provider object must still take fast direct GQA with an all-True mask.
        dense_provider = Provider(None)
        dense_fast_forward = llama_backend._SparseAttn(
            provider=dense_provider, fast=True
        )._make_forward(attention, layer_idx=0)
        llama_backend.repeat_kv = forbidden_repeat
        dense_fast = dense_fast_forward(
            hidden,
            position_embeddings=(None, None),
            past_key_values=cache,
        )[0]

        llama_backend.repeat_kv = original_repeat
        dense_slow_forward = llama_backend._SparseAttn(
            provider=dense_provider, fast=False
        )._make_forward(attention, layer_idx=0)
        dense_slow = dense_slow_forward(
            hidden,
            position_embeddings=(None, None),
            past_key_values=cache,
        )[0]
    finally:
        llama_backend.repeat_kv = original_repeat
        llama_backend.apply_rotary_pos_emb = original_rope

    torch.testing.assert_close(fast, slow, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(dense_fast, dense_slow, rtol=3e-5, atol=3e-6)


def _expect_empty_selection_error(allow) -> None:
    q, k, v = _case_tensors(8, 2, torch.float32, torch.device("cpu"))
    try:
        llama_backend._gather_sdpa(q, k, v, allow)
    except ValueError as exc:
        message = str(exc)
        assert "at least one token" in message, message
        assert "sink/recent" in message, message
    else:  # pragma: no cover - assertion helper
        raise AssertionError("zero selection must be rejected")


def test_zero_selection_is_rejected_explicitly() -> None:
    shared_empty = torch.zeros(1, 1, 1, T, dtype=torch.bool)
    _expect_empty_selection_error(shared_empty)

    one_query_head_empty = torch.zeros(1, 8, 1, T, dtype=torch.bool)
    one_query_head_empty[:, :, :, 0] = True
    one_query_head_empty[:, 3, :, :] = False
    _expect_empty_selection_error(one_query_head_empty)


def main() -> None:
    test_direct_gather_matches_full_width_reference()
    test_direct_gather_never_calls_repeat_kv()
    test_direct_gather_exposes_selected_positions_to_kv_transform()
    test_sparse_forward_fast_avoids_repeat_and_matches_slow()
    test_zero_selection_is_rejected_explicitly()
    print("all direct-GQA gather equivalence checks passed")


if __name__ == "__main__":
    main()
