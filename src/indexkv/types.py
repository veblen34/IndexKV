"""Core data types for the unified index-only method comparison.

The framework compares source-backed baseline adapters and evaluation
references on a single axis: given a layer's post-RoPE
keys/values, each method builds an index once at end-of-prefill and then
answers, for a decode query, which prompt positions should be attended to under
a fixed token budget.
Everything else (Q/K capture, mask building, generation, scoring, kernels,
offload, cache layout, scheduling) is shared or deliberately excluded so
numbers measure index quality rather than system engineering.

Reselection policy
------------------
* ``reselect="per_step"``  -> ``select`` is called at EVERY decode step with
  the current post-RoPE decode query ``(H_q, 1, D)``.
* ``reselect="static"``    -> a prefill-query slice is consumed when the
  budget-independent index is built and the selected mask is frozen for a
  budget during generation.

Two selection shapes are supported:

* ``kind="block"``      -> disjoint ``(start, end)`` ranges shared by all
  query heads.
* ``kind="per_head"``   -> token indices with H equal to 1, H_kv, or H_q.
  Ragged rows use an explicit boolean ``per_head_valid`` tensor; padding is
  never represented as a real token id.

The engine validates every selection against the normative fairness contract
in ``docs/fairness_contract.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch


QueryKind = Literal["last", "obs_window", "full_prefill"]
SelectionKind = Literal["block", "per_head"]
MethodScope = Literal["index_only", "dense_reference"]
MethodRole = Literal[
    "baseline",
    "evaluation_reference",
    "excluded_baseline",
]
ReselectPolicy = Literal["static", "per_step"]


@dataclass(frozen=True)
class QueryNeeds:
    """Declare the prefill inputs required by an index method.

    ``query`` controls the trailing prefill-Q slice made available to static
    index construction.  Per-step methods declare ``last`` but consume the
    live decode query instead.  ``value`` and ``weights_key`` request V or
    model-specific trained index tensors respectively.
    """

    query: QueryKind = "last"
    obs_window: int = 32
    value: bool = False
    weights_key: Optional[str] = None


@dataclass
class MethodConfig:
    """Shared run configuration plus method-specific values in ``extra``."""

    budget: int
    block_size: int = 32
    sink: int = 4
    recent: int = 32
    group_size: int = 1          # H_q // H_kv
    n_prompt: int = 0
    layer_idx: int = 0
    dense_prefix_layers: int = 0
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.budget < 0:
            raise ValueError(f"budget must be >= 0, got {self.budget}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.sink < 0 or self.recent < 0:
            raise ValueError(
                f"sink/recent must be >= 0, got sink={self.sink}, recent={self.recent}"
            )
        if self.group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {self.group_size}")
        if self.n_prompt < 0:
            raise ValueError(f"n_prompt must be >= 0, got {self.n_prompt}")
        if self.dense_prefix_layers < 0:
            raise ValueError(
                f"dense_prefix_layers must be >= 0, got {self.dense_prefix_layers}"
            )

    def get(self, key: str, default=None):
        return self.extra.get(key, default)


@dataclass
class LayerSelection:
    """A method's untrusted per-layer proposal before engine validation.

    Exactly one payload must be present.  ``per_head_valid`` is optional when
    every slot in ``per_head_idx`` is valid and required for ragged rows.
    """

    kind: SelectionKind
    blocks: Optional[torch.Tensor] = None
    per_head_idx: Optional[torch.Tensor] = None
    per_head_valid: Optional[torch.Tensor] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in ("block", "per_head"):
            raise ValueError(f"unknown selection kind {self.kind!r}")

        has_blocks = self.blocks is not None
        has_per_head = self.per_head_idx is not None
        if has_blocks == has_per_head:
            raise ValueError(
                "exactly one of `blocks` and `per_head_idx` must be provided"
            )

        if self.kind == "block":
            if not has_blocks:
                raise ValueError("block selection requires `blocks`")
            if self.per_head_valid is not None:
                raise ValueError("block selection cannot carry `per_head_valid`")
            if not isinstance(self.blocks, torch.Tensor):
                raise TypeError("blocks must be a torch.Tensor")
            if self.blocks.dtype != torch.long:
                raise TypeError(
                    f"blocks must have dtype torch.long, got {self.blocks.dtype}"
                )
            if self.blocks.ndim != 2 or self.blocks.shape[1] != 2:
                raise ValueError(
                    "blocks must have shape (num_blocks, 2), got "
                    f"{tuple(self.blocks.shape)}"
                )
            return

        if not has_per_head:
            raise ValueError("per_head selection requires `per_head_idx`")
        if not isinstance(self.per_head_idx, torch.Tensor):
            raise TypeError("per_head_idx must be a torch.Tensor")
        if self.per_head_idx.dtype != torch.long:
            raise TypeError(
                "per_head_idx must have dtype torch.long, got "
                f"{self.per_head_idx.dtype}"
            )
        if self.per_head_idx.ndim != 2:
            raise ValueError(
                "per_head_idx must have shape (heads, width), got "
                f"{tuple(self.per_head_idx.shape)}"
            )
        if self.per_head_valid is not None:
            if not isinstance(self.per_head_valid, torch.Tensor):
                raise TypeError("per_head_valid must be a torch.Tensor")
            if self.per_head_valid.dtype != torch.bool:
                raise TypeError(
                    "per_head_valid must have dtype torch.bool, got "
                    f"{self.per_head_valid.dtype}"
                )
            if self.per_head_valid.shape != self.per_head_idx.shape:
                raise ValueError(
                    "per_head_valid must have the same shape as per_head_idx, got "
                    f"{tuple(self.per_head_valid.shape)} vs "
                    f"{tuple(self.per_head_idx.shape)}"
                )
