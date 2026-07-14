"""Model-free smoke test: every implemented method through the real engine path.

Runs on CPU with random tensors. Validates, for each method:
  * build_indices -> make_provider -> mask_for round-trips,
  * mask shape/dtype, sink+recent always allowed,
  * per-head budgets respected (selected middle tokens <= budget per row,
    block methods <= budget rounded to their own chunk grid),
  * per-step methods reselect for every live query, while static masks freeze,
  * the same global dense prefix is enforced for every sparse method.

Usage:  python3 tests/test_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indexkv.backends.base import Capture, ModelDims
from indexkv.engine import build_indices, make_provider
from indexkv.registry import METHODS, get_method
import indexkv.methods  # noqa: F401  (populate registry)

H_KV, GROUP, D = 2, 2, 16
H_Q = H_KV * GROUP
N = 320
SINK, RECENT, BLOCK = 4, 32, 32
BUDGET = 64
LAYERS = [0, 1, 2]
DENSE_PREFIX = 2
RBITS = 32
HA_BITS = 8


class FakeBackend:
    dims = ModelDims(n_layers=len(LAYERS), n_heads=H_Q, n_kv_heads=H_KV, head_dim=D)


def fake_capture(gen):
    K = {L: torch.randn(H_KV, N, D, generator=gen) for L in LAYERS}
    Q = {L: torch.randn(H_Q, N, D, generator=gen) for L in LAYERS}
    V = {L: torch.randn(H_KV, N, D, generator=gen) for L in LAYERS}
    return Capture(K=K, Q=Q, V=V, n_prompt=N)


def fake_weights(name, gen):
    if name == "hata":
        return {
            L: torch.randn(H_KV, D, RBITS, generator=gen)
            for L in LAYERS[DENSE_PREFIX:]
        }
    if name == "hashattention":
        def mlp():
            return [
                (torch.randn(H_Q, 24, D, generator=gen),
                 torch.randn(H_Q, 24, generator=gen)),
                (torch.randn(H_Q, 24, 24, generator=gen),
                 torch.randn(H_Q, 24, generator=gen)),
                (torch.randn(H_Q, HA_BITS, 24, generator=gen),
                 torch.randn(H_Q, HA_BITS, generator=gen)),
            ]
        return {L: {"k": mlp(), "q": mlp()} for L in LAYERS}
    return None


def check_mask(name, mask, *, budget_cap):
    assert mask.dtype == torch.bool, f"{name}: mask dtype {mask.dtype}"
    assert mask.shape in ((1, 1, 1, N), (1, H_Q, 1, N)), f"{name}: shape {mask.shape}"
    flat = mask.reshape(-1, N)
    assert flat[:, :SINK].all(), f"{name}: sink not kept"
    assert flat[:, N - RECENT:].all(), f"{name}: recent not kept"
    mid = flat[:, SINK:N - RECENT]
    per_row = mid.sum(dim=-1)
    assert (per_row <= budget_cap).all(), (
        f"{name}: middle selection {per_row.tolist()} exceeds cap {budget_cap}"
    )
    assert (per_row > 0).all(), f"{name}: empty selection"


def main():
    gen = torch.Generator().manual_seed(0)
    backend = FakeBackend()
    cap = fake_capture(gen)
    q_step = torch.randn(H_Q, 1, D, generator=gen)
    q_step2 = torch.randn(H_Q, 1, D, generator=gen)

    common = dict(
        block_size=BLOCK, sink=SINK, recent=RECENT,
        dense_prefix_layers=DENSE_PREFIX,
        device="cpu", dtype=torch.float32,
    )

    to_test = [n for n, c in sorted(METHODS.items()) if c.implemented and n != "full"]
    print("testing:", to_test)

    for name in to_test:
        method = get_method(name)
        weights = fake_weights(name, gen)
        extra = {"avg_cluster_size": 8} if name == "wave_index" else None

        indices = build_indices(backend, method, cap, weights=weights,
                                extra=extra, **common)
        provider = make_provider(backend, method, cap, indices, BUDGET,
                                 weights=weights, extra=extra, **common)

        # The index budget covers only the eligible middle region. Atomic
        # blocks/chunks/clusters may under-use it but can never exceed it.
        budget_cap = BUDGET
        for L in LAYERS:
            m = provider.mask_for(L, q_step)
            if L < DENSE_PREFIX:
                assert m is None, (
                    f"{name}: global dense-prefix layer {L} returned a mask"
                )
                assert indices[L] is None
                continue
            assert m is not None, f"{name}: layer {L} unexpectedly dense"
            check_mask(f"{name}[L{L}]", m, budget_cap=budget_cap)

        m1 = provider.mask_for(LAYERS[-1], q_step)
        m2 = provider.mask_for(LAYERS[-1], q_step2)
        assert m1.shape == m2.shape
        check_mask(f"{name}[q2]", m2, budget_cap=budget_cap)
        if method.reselect == "static":
            assert torch.equal(m1, m2), f"{name}: static mask changed"
        print(f"  ok {name:14s} kind={method.kind:8s} reselect={method.reselect}")

    # full through the engine's static path (sanity for the mask fallback)
    full = get_method("full")
    indices = build_indices(backend, full, cap, **common)
    provider = make_provider(backend, full, cap, indices, 0, **common)
    assert provider.mask_for(0, q_step) is None
    assert provider.mask_for(1, q_step) is None
    m = provider.mask_for(2, q_step)
    assert m.all(), "full must keep everything on sparse-policy layers"
    print("  ok full           (mask fallback keeps everything)")
    print("all synthetic checks passed")


if __name__ == "__main__":
    main()
