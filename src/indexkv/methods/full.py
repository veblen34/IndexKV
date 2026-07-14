"""Dense reference — no sparsity. Validates the whole pipeline end-to-end.

``full`` is special-cased by the runner (it passes ``provider=None`` so the
backend uses its default dense attention). It is registered here mainly so it
appears in listings and so a mask-based sanity path exists: if ever selected
through the engine, it keeps every non-sink/recent block, i.e. the entire
prompt.
"""

from __future__ import annotations

import torch

from ..masks import make_blocks
from ..registry import IndexMethod, register
from ..types import LayerSelection, MethodConfig, QueryNeeds


@register(
    "full",
    kind="block",
    needs=QueryNeeds(query="last"),
    reselect="static",
    scope="dense_reference",
    reference="dense reference (no index; attends to all causal KV)",
)
class Full(IndexMethod):
    def build(self, K, V, cfg: MethodConfig, Q=None):
        # No index needed; keep N for block construction.
        return K.shape[1]

    def select(self, index, Q, cfg: MethodConfig) -> LayerSelection:
        n = index
        blocks = make_blocks(n, cfg.block_size, sink=cfg.sink, recent=cfg.recent)
        return LayerSelection(kind="block", blocks=blocks.to(cfg.device))
