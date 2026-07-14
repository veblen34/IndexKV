"""Shared sweep plumbing for index-only benchmarks.

One method instance and one budget-independent index are reused across all
budgets for a sample.  The runner also owns the global dense-prefix policy,
trained-index weight loading, synchronized build timing, and the latest
selection statistics exposed by providers.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch

from .backends.base import Capture, ModelBackend
from .engine import DenseProvider, build_indices, make_provider
from .registry import METHODS, get_method, list_methods
from .weights import WeightsError, load_method_weights


TIMING_SEMANTICS_VERSION = "indexkv-exclusive-timing-v1"

_SELECTION_SCOPES = {"none", "provider_setup", "decode"}


def _partition_decode_timing(
    decode_inclusive_s: float,
    index_query_s: float,
    selection_scope: str,
) -> tuple[float, float]:
    """Return exclusive model-decode time and decomposition error.

    ``decode_inclusive_s`` and ``index_query_s`` must use the same clock. On
    CUDA they are outer/nested events on the same device and current stream;
    on CPU they are outer/nested ``perf_counter`` intervals. Static selection
    lives outside decode and therefore is not subtracted.
    """
    decode_inclusive_s = float(decode_inclusive_s)
    index_query_s = float(index_query_s)
    if selection_scope not in _SELECTION_SCOPES:
        raise ValueError(f"unknown selection scope {selection_scope!r}")
    for name, value in (
        ("decode_inclusive_s", decode_inclusive_s),
        ("index_query_s", index_query_s),
    ):
        if not torch.isfinite(torch.tensor(value)) or value < 0:
            raise ValueError(f"{name} must be finite and >= 0, got {value}")
    if selection_scope == "none" and index_query_s != 0:
        raise RuntimeError("dense decode reported nonzero index-query time")
    nested_query_s = index_query_s if selection_scope == "decode" else 0.0
    exclusive_s = decode_inclusive_s - nested_query_s
    tolerance = max(1e-9, decode_inclusive_s * 1e-6)
    if exclusive_s < -tolerance:
        raise RuntimeError(
            "nested index-query time exceeds its inclusive decode interval: "
            f"query={index_query_s:.9f}s decode={decode_inclusive_s:.9f}s"
        )
    exclusive_s = max(exclusive_s, 0.0)
    error_s = abs(decode_inclusive_s - (exclusive_s + nested_query_s))
    return exclusive_s, error_s


def _parse_value(value: str):
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def parse_set_overrides(pairs: List[str]) -> Dict[str, dict]:
    """Parse and validate ``--set method.knob=value`` overrides."""
    out: Dict[str, dict] = {}
    for item in pairs:
        if "=" not in item:
            raise SystemExit(
                f"--set expects method.knob=value (missing '='): {item!r}"
            )
        key, value = item.split("=", 1)
        if "." not in key:
            raise SystemExit(
                f"--set expects method.knob=value (missing method): {item!r}"
            )
        method, knob = key.split(".", 1)
        if not method or not knob:
            raise SystemExit(f"--set expects method.knob=value, got {item!r}")
        if method not in METHODS:
            raise SystemExit(
                f"--set targets unknown method {method!r}; known: {sorted(METHODS)}"
            )
        out.setdefault(method, {})[knob] = _parse_value(value)
    return out


def print_method_listing():
    for name, meta in list_methods().items():
        scope = "" if meta["scope"] == "index_only" else f" scope={meta['scope']}"
        print(
            f"  {name:14s} kind={meta['kind']:8s} "
            f"reselect={meta['reselect']:8s}{scope}"
        )
        print(f"      {meta['reference']}")


class SweepRunner:
    """Hold the backend and all per-model method state for a fair sweep."""

    def __init__(
        self,
        backend: ModelBackend,
        method_names: List[str],
        *,
        model_name: str,
        block_size: int = 32,
        sink: int = 4,
        recent: int = 32,
        dense_prefix_layers: int = 2,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        overrides: Optional[Dict[str, dict]] = None,
    ):
        if not method_names:
            raise ValueError("method_names must contain at least one method")
        seen = set()
        duplicates = set()
        for name in method_names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        if duplicates:
            raise ValueError(
                f"method_names must be unique, duplicates={sorted(duplicates)}"
            )
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        if sink < 0 or recent < 0:
            raise ValueError(
                f"sink/recent must be >= 0, got sink={sink}, recent={recent}"
            )
        if not 0 <= dense_prefix_layers <= backend.dims.n_layers:
            raise ValueError(
                "dense_prefix_layers must be in [0, n_layers], got "
                f"{dense_prefix_layers} for n_layers={backend.dims.n_layers}"
            )

        self.backend = backend
        self.block_size = block_size
        self.sink = sink
        self.recent = recent
        self.dense_prefix_layers = dense_prefix_layers
        self.device = device
        self.dtype = dtype
        self.overrides = overrides or {}
        unknown_overrides = set(self.overrides) - set(method_names)
        if unknown_overrides:
            raise ValueError(
                "overrides supplied for methods outside this sweep: "
                f"{sorted(unknown_overrides)}"
            )

        self.runnable: List[str] = []
        self.skipped: Dict[str, str] = {}
        self.weights: Dict[str, Dict[int, object]] = {}
        self.methods: Dict[str, object] = {}
        self.last_selection_stats: dict = {}
        self.last_generation_timing: dict = {}

        for name in method_names:
            if name not in METHODS:
                raise SystemExit(
                    f"unknown method '{name}'. Known: {sorted(METHODS)}"
                )
            meta = METHODS[name]
            if not meta.implemented:
                self.skipped[name] = f"not-implemented: {meta.reference}"
                continue
            wk = meta.needs.weights_key
            if wk is not None:
                try:
                    self.weights[name] = load_method_weights(
                        wk,
                        model_name,
                        device=device,
                        dtype=dtype,
                        extra=self.overrides.get(name),
                    )
                except (WeightsError, OSError, RuntimeError, KeyError, ValueError) as e:
                    self.skipped[name] = f"weights-missing: {e}"
                    continue
            self.methods[name] = get_method(name)
            self.runnable.append(name)

        if not self.runnable:
            detail = ", ".join(
                f"{name}: {reason}" for name, reason in self.skipped.items()
            ) or "no implemented methods"
            raise ValueError(f"no runnable methods remain after validation ({detail})")
        self.need_value = any(
            METHODS[m].needs.value for m in self.runnable
        )
        self.q_window = self._compute_q_window()

    def _compute_q_window(self):
        """Maximum trailing prefill-Q slice needed by any static method."""
        W = 0
        for name in self.runnable:
            if name == "full":
                continue
            method = METHODS[name]
            if method.reselect != "static":
                continue
            query_kind = method.needs.query
            if query_kind == "full_prefill":
                return None
            W = max(
                W,
                1
                if query_kind == "last"
                else max(1, method.needs.obs_window),
            )
        return W

    def capture(self, ids: torch.Tensor) -> Capture:
        if not any(name != "full" for name in self.runnable):
            return Capture(K={}, Q={}, V=None, n_prompt=int(ids.shape[1]))
        return self.backend.capture(
            ids,
            need_value=self.need_value,
            dtype=self.dtype,
            q_window=self.q_window,
        )

    def build(self, method_name: str, cap: Capture):
        """Build one method index once and return (indices, synchronized seconds)."""
        if method_name == "full":
            return None, 0.0
        cuda_device = next(
            (
                tensor.device
                for tensor in cap.K.values()
                if tensor.device.type == "cuda"
            ),
            None,
        )
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        t0 = time.perf_counter()
        idx = build_indices(
            self.backend,
            self.methods[method_name],
            cap,
            block_size=self.block_size,
            sink=self.sink,
            recent=self.recent,
            dense_prefix_layers=self.dense_prefix_layers,
            device=self.device,
            dtype=self.dtype,
            weights=self.weights.get(method_name),
            extra=self.overrides.get(method_name),
        )
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        return idx, time.perf_counter() - t0

    def generate(
        self,
        method_name: str,
        indices,
        cap: Capture,
        budget: Optional[int],
        ids: torch.Tensor,
        max_new: int,
    ) -> str:
        self.last_selection_stats = {}
        self.last_generation_timing = {}
        cuda_device = ids.device if ids.device.type == "cuda" else None

        def synchronize() -> None:
            if cuda_device is not None:
                torch.cuda.synchronize(cuda_device)

        synchronize()
        total_start = time.perf_counter()
        setup_start = total_start
        if method_name == "full":
            selection_scope = "none"
            provider = DenseProvider()
        else:
            selection_scope = (
                "provider_setup"
                if self.methods[method_name].reselect == "static"
                else "decode"
            )
            if budget is None:
                raise ValueError(f"sparse method {method_name!r} requires a budget")
            provider = make_provider(
                self.backend,
                self.methods[method_name],
                cap,
                indices,
                budget,
                block_size=self.block_size,
                sink=self.sink,
                recent=self.recent,
                dense_prefix_layers=self.dense_prefix_layers,
                device=self.device,
                dtype=self.dtype,
                weights=self.weights.get(method_name),
                extra=self.overrides.get(method_name),
            )
        synchronize()
        setup_end = time.perf_counter()

        decode_wall_start = time.perf_counter()
        decode_start_event = None
        decode_end_event = None
        if cuda_device is not None:
            with torch.cuda.device(cuda_device):
                decode_start_event = torch.cuda.Event(enable_timing=True)
                decode_start_event.record()
        text = self.backend.sparse_generate(ids, max_new, provider)
        if cuda_device is not None:
            with torch.cuda.device(cuda_device):
                decode_end_event = torch.cuda.Event(enable_timing=True)
                decode_end_event.record()
        synchronize()
        generate_end = time.perf_counter()
        decode_wall_s = generate_end - decode_wall_start
        if cuda_device is None:
            decode_clock = "cpu_wall"
            decode_inclusive_s = decode_wall_s
        else:
            decode_clock = "cuda_event"
            decode_inclusive_s = decode_start_event.elapsed_time(decode_end_event) / 1000.0

        if hasattr(provider, "stats"):
            self.last_selection_stats = provider.stats()
        selection_s_total = float(
            self.last_selection_stats.get("selection_s_total", 0.0)
        )
        model_decode_s, decomposition_error_s = _partition_decode_timing(
            decode_inclusive_s,
            selection_s_total,
            selection_scope,
        )
        self.last_generation_timing = {
            "timing_semantics": TIMING_SEMANTICS_VERSION,
            "decode_clock": decode_clock,
            "selection_scope": selection_scope,
            "provider_setup_wall_s": setup_end - setup_start,
            "decode_wall_s_inclusive": decode_wall_s,
            "e2e_wall_s": generate_end - total_start,
            "decode_s_inclusive": decode_inclusive_s,
            "model_decode_s_exclusive": model_decode_s,
            "decomposition_error_s": decomposition_error_s,
            # Backward-compatible raw-wall aliases; paper records use the
            # explicitly named fields above.
            "provider_setup_s": setup_end - setup_start,
            "backend_generate_s": decode_wall_s,
            "total_s": generate_end - total_start,
            "selection_s_total": selection_s_total,
        }
        return text
