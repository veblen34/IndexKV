"""Selection engine enforcing the index-only fairness contract.

The engine owns lifecycle, global dense-layer policy, query slicing, strict
selection validation, and conversion to the one shared backend mask format.
Method code never receives a model object and cannot silently change the common
budget/window policy.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import torch

from .backends.base import Capture, ModelBackend, SelectionProvider
from .masks import selection_to_mask
from .registry import IndexMethod
from .types import MethodConfig


def _slice_query(Q_full: torch.Tensor, method: IndexMethod) -> torch.Tensor:
    """Return exactly the trailing prefill-Q slice declared by a static method."""
    needs = method.needs
    if Q_full.ndim != 3:
        raise ValueError(
            f"captured Q must have shape (H_q, positions, D), got {tuple(Q_full.shape)}"
        )
    if needs.query == "last":
        return Q_full[:, -1:, :]
    if needs.query == "obs_window":
        t = max(1, needs.obs_window)
        return Q_full[:, -t:, :]
    if needs.query == "full_prefill":
        return Q_full
    raise ValueError(needs.query)


def _validate_dense_prefix(backend: ModelBackend, dense_prefix_layers: int):
    if dense_prefix_layers < 0 or dense_prefix_layers > backend.dims.n_layers:
        raise ValueError(
            "dense_prefix_layers must be in [0, n_layers], got "
            f"{dense_prefix_layers} for n_layers={backend.dims.n_layers}"
        )


def _make_cfg(
    backend: ModelBackend,
    layer_idx: int,
    n_prompt: int,
    budget: int,
    *,
    block_size: int,
    sink: int,
    recent: int,
    dense_prefix_layers: int,
    device: str,
    dtype: torch.dtype,
    weights: Optional[Dict[int, object]],
    extra: Optional[dict],
) -> MethodConfig:
    cfg = MethodConfig(
        budget=budget,
        block_size=block_size,
        sink=sink,
        recent=recent,
        group_size=backend.dims.group_size,
        n_prompt=n_prompt,
        layer_idx=layer_idx,
        dense_prefix_layers=dense_prefix_layers,
        device=device,
        dtype=dtype,
        extra=dict(extra or {}),
    )
    if weights is not None and layer_idx in weights:
        cfg.extra["weights"] = weights[layer_idx]
    return cfg


@torch.no_grad()
def build_indices(
    backend: ModelBackend,
    method: IndexMethod,
    capture: Capture,
    *,
    block_size: int = 32,
    sink: int = 4,
    recent: int = 32,
    dense_prefix_layers: int = 0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    weights: Optional[Dict[int, object]] = None,
    extra: Optional[dict] = None,
) -> Dict[int, object]:
    """Build one budget-independent index per captured layer.

    Global dense-prefix layers are represented by ``None`` for every method.
    Static methods receive their declared prefill-query slice here so expensive
    budget-independent scoring is not repeated for every budget.
    """
    _validate_dense_prefix(backend, dense_prefix_layers)
    indices: Dict[int, object] = {}
    for L, K in capture.K.items():
        if L < dense_prefix_layers:
            indices[L] = None
            continue
        V = (
            capture.V[L]
            if capture.V is not None and method.needs.value
            else None
        )
        Q_build = None
        if method.reselect == "static" and method.scope != "dense_reference":
            if L not in capture.Q:
                raise ValueError(
                    f"static method {method.name!r} needs captured Q for layer {L}"
                )
            Q_build = _slice_query(capture.Q[L], method)
            if Q_build.shape[1] == 0:
                raise ValueError(
                    f"static method {method.name!r} received an empty Q slice"
                )
        cfg = _make_cfg(
            backend,
            L,
            capture.n_prompt,
            0,
            block_size=block_size,
            sink=sink,
            recent=recent,
            dense_prefix_layers=dense_prefix_layers,
            device=device,
            dtype=dtype,
            weights=weights,
            extra=extra,
        )
        indices[L] = method.build(K, V, cfg, Q=Q_build)
    return indices


class _SelectionTimer:
    """Accumulate provider work without synchronizing inside decode callbacks."""

    def __init__(self):
        self._cpu_s = 0.0
        self._cuda_s = 0.0
        self._cuda_events: list[tuple[torch.device, object, object]] = []

    def start(self, device: torch.device | str):
        device = torch.device(device)
        if device.type != "cuda":
            return ("cpu", time.perf_counter())
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA selection timing requested but CUDA is unavailable")
        with torch.cuda.device(device):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
        return ("cuda", device, start)

    def stop(self, token) -> None:
        if token[0] == "cpu":
            self._cpu_s += time.perf_counter() - token[1]
            return
        _, device, start = token
        with torch.cuda.device(device):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
        self._cuda_events.append((device, start, end))

    def total_s(self) -> float:
        """Resolve all pending CUDA pairs after one synchronize per device."""
        if self._cuda_events:
            devices = sorted(
                {device for device, _, _ in self._cuda_events},
                key=str,
            )
            for device in devices:
                torch.cuda.synchronize(device)
            self._cuda_s += sum(
                start.elapsed_time(end)
                for _, start, end in self._cuda_events
            ) / 1000.0
            self._cuda_events.clear()
        return self._cpu_s + self._cuda_s


class _StatsMixin:
    """Constant-space provider audit statistics with deferred device readback."""

    def _init_stats(self, selection_timer: Optional[_SelectionTimer] = None):
        self.selection_calls: Dict[int, int] = {}
        self._selection_aggregate: Dict[int, Dict[str, torch.Tensor]] = {}
        self._selection_timer = selection_timer or _SelectionTimer()

    def _record_counts(self, layer_idx: int, counts: torch.Tensor) -> None:
        self.selection_calls[layer_idx] = self.selection_calls.get(layer_idx, 0) + 1
        counts = counts.detach().reshape(-1).to(dtype=torch.long)
        if counts.numel() == 0:
            return
        update = {
            "sample_count": torch.full(
                (), counts.numel(), dtype=torch.long, device=counts.device
            ),
            "sum": counts.sum(),
            "min": counts.min(),
            "max": counts.max(),
        }
        current = self._selection_aggregate.get(layer_idx)
        if current is None:
            self._selection_aggregate[layer_idx] = update
            return
        current["sample_count"].add_(update["sample_count"])
        current["sum"].add_(update["sum"])
        current["min"] = torch.minimum(current["min"], update["min"])
        current["max"] = torch.maximum(current["max"], update["max"])

    def _record(
        self,
        layer_idx: int,
        mask: torch.Tensor,
        n_prompt: int,
        sink: int,
        recent: int,
        H_q: Optional[int] = None,
    ) -> None:
        flat = mask.reshape(-1, n_prompt)
        lo = min(sink, n_prompt)
        hi = max(lo, n_prompt - recent)
        counts = flat[:, lo:hi].sum(dim=-1)
        if counts.numel() == 1 and H_q is not None and H_q > 1:
            counts = counts.expand(H_q)
        self._record_counts(layer_idx, counts)

    def stats(self) -> dict:
        # Resolve selection events first. The following aggregate transfer is
        # then one compact D2H copy per device, never one copy per call/head.
        selection_s_total = self._selection_timer.total_s()
        aggregate: Dict[int, dict] = {}
        by_device: Dict[torch.device, list[tuple[int, Dict[str, torch.Tensor]]]] = {}
        for layer_idx, values in self._selection_aggregate.items():
            device = values["sum"].device
            by_device.setdefault(device, []).append((layer_idx, values))
        for _, entries in by_device.items():
            entries.sort(key=lambda item: item[0])
            packed = torch.stack([
                torch.stack([
                    values["sample_count"],
                    values["sum"],
                    values["min"],
                    values["max"],
                ])
                for _, values in entries
            ])
            host_values = packed.to(device="cpu").tolist()
            for (layer_idx, _), row in zip(entries, host_values):
                aggregate[int(layer_idx)] = {
                    "sample_count": int(row[0]),
                    "sum": int(row[1]),
                    "min": int(row[2]),
                    "max": int(row[3]),
                }
        return {
            "selection_calls": dict(self.selection_calls),
            "selection_aggregate": aggregate,
            # Compatibility key for old consumers; traces are intentionally gone.
            "selected_counts": {},
            "selection_s_total": float(selection_s_total),
        }


class StaticProvider(_StatsMixin, SelectionProvider):
    """Frozen masks whose one-time selection is included in provider timing."""

    def __init__(
        self,
        masks: Dict[int, Optional[torch.Tensor]],
        *,
        n_prompt: Optional[int] = None,
        sink: int = 0,
        recent: int = 0,
        H_q: Optional[int] = None,
        selection_timer: Optional[_SelectionTimer] = None,
        selected_counts: Optional[Dict[int, list[int]]] = None,
    ):
        self._masks = masks
        self._init_stats(selection_timer)
        # selected_counts remains accepted for compatibility with older callers.
        if selected_counts is not None:
            for layer_idx, counts in selected_counts.items():
                self._record_counts(layer_idx, torch.as_tensor(counts))
        elif n_prompt is not None:
            for layer_idx, mask in masks.items():
                if mask is not None:
                    self._record(layer_idx, mask, n_prompt, sink, recent, H_q)

    def mask_for(self, layer_idx: int, q: torch.Tensor) -> Optional[torch.Tensor]:
        return self._masks.get(layer_idx)


class DenseProvider(SelectionProvider):
    """Request the shared patched attention path without restricting KV.

    A non-``None`` provider makes the backend attach the same attention
    implementation used by indexed methods. Returning ``None`` per layer
    leaves every causal KV position visible, giving a dense reference without
    falling back to a different HuggingFace decode implementation.
    """

    def mask_for(self, layer_idx: int, q: torch.Tensor) -> None:
        return None

    def stats(self) -> dict:
        return {
            "selection_calls": {},
            "selection_aggregate": {},
            "selected_counts": {},
            "selection_s_total": 0.0,
        }


class PerStepProvider(_StatsMixin, SelectionProvider):
    """Re-run method.select with the live decode query at every sparse step."""

    def __init__(
        self,
        method: IndexMethod,
        indices: Dict[int, object],
        cfgs: Dict[int, MethodConfig],
        n_prompt: int,
        H_q: int,
        group_size: int,
        sink: int,
        recent: int,
    ):
        self.method = method
        self.indices = indices
        self.cfgs = cfgs
        self.n_prompt = n_prompt
        self.H_q = H_q
        self.group_size = group_size
        self.sink = sink
        self.recent = recent
        self._init_stats()

    @torch.no_grad()
    def mask_for(self, layer_idx: int, q: torch.Tensor) -> Optional[torch.Tensor]:
        index = self.indices.get(layer_idx)
        if index is None:
            return None
        cfg = self.cfgs[layer_idx]
        timing = self._selection_timer.start(q.device)
        try:
            sel = self.method.select(index, q, cfg)
            mask = selection_to_mask(
                sel,
                expected_kind=self.method.kind,
                n_prompt=self.n_prompt,
                budget=cfg.budget,
                sink=self.sink,
                recent=self.recent,
                H_q=self.H_q,
                group_size=self.group_size,
                device=q.device,
            )
            self._record(
                layer_idx, mask, self.n_prompt, self.sink, self.recent, self.H_q
            )
        finally:
            self._selection_timer.stop(timing)
        return mask

    def has_kv_transform(self, layer_idx: int) -> bool:
        index = self.indices.get(layer_idx)
        return bool(
            index is not None
            and self.method.has_kv_transform(index, self.cfgs[layer_idx])
        )

    @torch.no_grad()
    def transform_selected_kv(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.indices.get(layer_idx)
        if index is None:
            return k, v
        return self.method.transform_selected_kv(
            index, k, v, positions, self.cfgs[layer_idx]
        )


@torch.no_grad()
def make_provider(
    backend: ModelBackend,
    method: IndexMethod,
    capture: Capture,
    indices: Dict[int, object],
    budget: int,
    *,
    block_size: int = 32,
    sink: int = 4,
    recent: int = 32,
    dense_prefix_layers: int = 0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    weights: Optional[Dict[int, object]] = None,
    extra: Optional[dict] = None,
) -> SelectionProvider:
    """Create a budget-specific provider from one shared built index."""
    _validate_dense_prefix(backend, dense_prefix_layers)
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    dims = backend.dims
    n_prompt = capture.n_prompt
    cfgs = {
        L: _make_cfg(
            backend,
            L,
            n_prompt,
            budget,
            block_size=block_size,
            sink=sink,
            recent=recent,
            dense_prefix_layers=dense_prefix_layers,
            device=device,
            dtype=dtype,
            weights=weights,
            extra=extra,
        )
        for L in indices
    }

    if method.reselect == "per_step":
        return PerStepProvider(
            method,
            indices,
            cfgs,
            n_prompt=n_prompt,
            H_q=dims.n_heads,
            group_size=dims.group_size,
            sink=sink,
            recent=recent,
        )

    masks: Dict[int, Optional[torch.Tensor]] = {}
    selection_timer = _SelectionTimer()
    for L, index in indices.items():
        if index is None:
            masks[L] = None
            continue
        cfg = cfgs[L]
        validation_budget = (
            n_prompt if method.scope == "dense_reference" else cfg.budget
        )
        layer_device = (
            capture.K[L].device if L in capture.K else torch.device(device)
        )
        timing = selection_timer.start(layer_device)
        try:
            sel = method.select(index, None, cfg)
            mask = selection_to_mask(
                sel,
                expected_kind=method.kind,
                n_prompt=n_prompt,
                budget=validation_budget,
                sink=sink,
                recent=recent,
                H_q=dims.n_heads,
                group_size=dims.group_size,
                device=layer_device,
            )
        finally:
            selection_timer.stop(timing)
        masks[L] = mask
    return StaticProvider(
        masks,
        n_prompt=n_prompt,
        sink=sink,
        recent=recent,
        H_q=dims.n_heads,
        selection_timer=selection_timer,
    )
