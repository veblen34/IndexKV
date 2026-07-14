"""Shared model-backend abstraction for the unified comparison.

Q/K capture and sparse decode live behind :class:`ModelBackend` so every
baseline adapter uses the same model execution.
Method code never imports ``transformers`` or installs a private model path.

A backend provides three things:

1. ``dims`` — head counts / head_dim / layer count read from the model config.
2. ``capture(prompt_ids)`` — run one prefill pass and return per-layer post-RoPE
   K (and Q, and optionally V) for index construction.
3. ``sparse_generate(prompt_ids, max_new, provider)`` — greedy decode that asks
   the :class:`SelectionProvider` for each layer's allow-mask at every step,
   passing the live post-RoPE decode query so per-step methods can re-select.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class ModelDims:
    n_layers: int
    n_heads: int          # H_q
    n_kv_heads: int       # H_kv
    head_dim: int

    def __post_init__(self) -> None:
        fields = {
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
        }
        for name, value in fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"{name} must be a positive integer, got {value!r}"
                )
        if self.n_heads % self.n_kv_heads:
            raise ValueError(
                f"n_heads={self.n_heads} must be divisible by "
                f"n_kv_heads={self.n_kv_heads}"
            )

    @property
    def group_size(self) -> int:
        return self.n_heads // self.n_kv_heads


@dataclass
class Capture:
    """Per-layer post-RoPE tensors captured during prefill.

    Each dict maps layer_idx -> tensor on the target device.

    K : (H_kv, N, D)
    Q : (H_q,  N, D)   full-prefill queries; sliced per static method's needs
    V : (H_kv, N, D)   only populated when some method needs values
    """

    K: Dict[int, torch.Tensor]
    Q: Dict[int, torch.Tensor]
    V: Optional[Dict[int, torch.Tensor]] = None
    n_prompt: int = 0


class SelectionProvider:
    """Maps (layer, live decode query) -> allow-mask during sparse decode.

    ``q`` is the current step's post-RoPE query, shape (H_q, 1, D). The
    returned mask is a bool tensor broadcastable to (1, H_q, 1, n_prompt), or
    ``None`` for a dense layer. Static methods ignore ``q`` and return their
    frozen mask; per-step methods re-run selection against it.
    """

    def mask_for(self, layer_idx: int, q: torch.Tensor) -> Optional[torch.Tensor]:
        raise NotImplementedError


class ModelBackend:
    """Interface every model backend implements."""

    dims: ModelDims

    def tokenize(self, prompt: str, *, chat_template: bool = False) -> torch.Tensor:
        """Turn a prompt string into (1, N) input ids on the model device.

        ``chat_template=True`` wraps the prompt as a single user turn and adds
        the generation prompt (LongBench v2 protocol — official ``pred.py`` calls
        the chat endpoint). ``chat_template=False`` tokenizes verbatim (RULER
        protocol — the ``input`` field is already the full prompt).
        """
        raise NotImplementedError

    def capture(self, prompt_ids: torch.Tensor, *, need_value: bool = False,
                dtype: torch.dtype = torch.bfloat16,
                q_window: Optional[int] = None) -> Capture:
        """Capture one batch-one prefill.

        ``q_window`` is the trailing query count to retain: ``None`` keeps all
        queries and ``0`` keeps none. Runners compute it from the sweep methods,
        so every backend must implement it rather than silently ignore it.
        """
        raise NotImplementedError

    def sparse_generate(
        self,
        prompt_ids: torch.Tensor,
        max_new: int,
        provider: Optional[SelectionProvider],
    ) -> str:
        raise NotImplementedError
