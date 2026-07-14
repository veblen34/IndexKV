"""Validate method selections and convert them into prompt allow-masks.

The validation in this module is part of the experiment protocol, not merely
defensive programming.  A method may select only unique positions in the common
eligible middle region and may never exceed the per-query-head index budget.
Invalid selections fail fast; indices are never silently clamped.
"""

from __future__ import annotations

from typing import Optional

import torch

from .types import LayerSelection, SelectionKind


def _effective_regions(n_prompt: int, sink: int, recent: int) -> tuple[int, int]:
    if n_prompt < 0:
        raise ValueError(f"n_prompt must be >= 0, got {n_prompt}")
    if sink < 0 or recent < 0:
        raise ValueError(
            f"sink/recent must be >= 0, got sink={sink}, recent={recent}"
        )
    lo = min(sink, n_prompt)
    hi = max(lo, n_prompt - recent)
    return lo, hi


@torch.no_grad()
def make_blocks(
    n: int, block_size: int, sink: int = 0, recent: int = 0
) -> torch.Tensor:
    """Partition the common eligible region into contiguous ranges."""
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    lo, hi = _effective_regions(n, sink, recent)
    starts = list(range(lo, hi, block_size))
    ends = [min(s + block_size, hi) for s in starts]
    if not starts:
        return torch.empty(0, 2, dtype=torch.long)
    return torch.tensor(list(zip(starts, ends)), dtype=torch.long)


@torch.no_grad()
def validate_selection(
    sel: LayerSelection,
    *,
    expected_kind: SelectionKind,
    n_prompt: int,
    budget: int,
    sink: int,
    recent: int,
    H_q: int,
    group_size: int,
) -> LayerSelection:
    """Validate a method proposal against the normative fairness contract.

    The returned object is the original selection.  Validation is intentionally
    strict so method bugs cannot be hidden by mask broadcasting, clamping, or
    duplicate padding.
    """
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    if sel.kind != expected_kind:
        raise ValueError(
            f"method declared kind={expected_kind!r} but returned {sel.kind!r}"
        )
    if H_q <= 0 or group_size <= 0 or H_q % group_size:
        raise ValueError(
            f"invalid GQA dimensions H_q={H_q}, group_size={group_size}"
        )
    H_kv = H_q // group_size
    lo, hi = _effective_regions(n_prompt, sink, recent)

    if sel.kind == "block":
        rows = sel.blocks.detach().cpu().tolist()
        ordered = sorted((int(s), int(e)) for s, e in rows)
        total = 0
        prev_end: Optional[int] = None
        for s, e in ordered:
            if not (s < e):
                raise ValueError(f"block must satisfy start < end, got [{s}, {e})")
            if s < lo or e > hi:
                raise ValueError(
                    f"block [{s}, {e}) is outside eligible region [{lo}, {hi})"
                )
            if prev_end is not None and s < prev_end:
                raise ValueError("selected blocks must be disjoint")
            total += e - s
            prev_end = e
        if total > budget:
            raise ValueError(
                f"block selection uses {total} unique tokens, budget is {budget}"
            )
        return sel

    idx = sel.per_head_idx
    valid = (
        torch.ones_like(idx, dtype=torch.bool)
        if sel.per_head_valid is None
        else sel.per_head_valid.to(device=idx.device)
    )
    H_in = idx.shape[0]
    if H_in not in (1, H_kv, H_q):
        raise ValueError(
            f"per_head selection rows must be 1, H_kv={H_kv}, or H_q={H_q}; "
            f"got {H_in}"
        )
    if sel.per_head_valid is not None:
        invalid_values = idx.masked_select(~valid)
        if invalid_values.numel() and not bool((invalid_values == -1).all()):
            raise ValueError("invalid ragged slots must use sentinel index -1")

    for h in range(H_in):
        ids = idx[h].masked_select(valid[h])
        if ids.numel() > budget:
            raise ValueError(
                f"selection row {h} has {ids.numel()} tokens, budget is {budget}"
            )
        if ids.numel() == 0:
            continue
        if bool(((ids < lo) | (ids >= hi)).any()):
            bad = ids[((ids < lo) | (ids >= hi))][0].item()
            raise ValueError(
                f"selected index {bad} is outside eligible region [{lo}, {hi})"
            )
        if ids.unique().numel() != ids.numel():
            raise ValueError(f"selection row {h} contains duplicate valid indices")
    return sel


@torch.no_grad()
def build_allow_mask_keys(
    chosen_blocks: torch.Tensor,
    n_prompt: int,
    sink: int,
    recent: int,
    device,
) -> torch.Tensor:
    """Validated block ranges -> shared (1, 1, 1, n_prompt) allow-mask."""
    lo, hi = _effective_regions(n_prompt, sink, recent)
    keep = torch.zeros(n_prompt, dtype=torch.bool, device=device)
    keep[:lo] = True
    keep[hi:] = True
    for s, e in chosen_blocks.detach().cpu().tolist():
        keep[int(s):int(e)] = True
    return keep.view(1, 1, 1, -1)


@torch.no_grad()
def build_per_head_allow_mask(
    per_head_idx: torch.Tensor,
    per_head_valid: Optional[torch.Tensor],
    n_prompt: int,
    sink: int,
    recent: int,
    H_q: int,
    group_size: int,
    device,
) -> torch.Tensor:
    """Validated shared/KV-head/query-head ids -> query-head allow-mask."""
    lo, hi = _effective_regions(n_prompt, sink, recent)
    idx = per_head_idx.to(device)
    valid = (
        torch.ones_like(idx, dtype=torch.bool, device=device)
        if per_head_valid is None
        else per_head_valid.to(device=device)
    )
    H_in = idx.shape[0]
    H_kv = H_q // group_size
    selected = torch.zeros(H_in, n_prompt, dtype=torch.bool, device=device)
    if bool(valid.any()):
        safe_idx = idx.masked_fill(~valid, 0)
        counts = torch.zeros(
            H_in, n_prompt, dtype=torch.int32, device=device
        )
        counts.scatter_add_(1, safe_idx, valid.to(torch.int32))
        selected = counts > 0

    if H_in == 1:
        selected = selected.expand(H_q, n_prompt).clone()
    elif H_in == H_kv:
        selected = selected.repeat_interleave(group_size, dim=0)

    selected[:, :lo] = True
    selected[:, hi:] = True
    return selected.view(1, H_q, 1, n_prompt)


@torch.no_grad()
def selection_to_mask(
    sel: LayerSelection,
    *,
    expected_kind: SelectionKind,
    n_prompt: int,
    budget: int,
    sink: int,
    recent: int,
    H_q: int,
    group_size: int,
    device,
) -> torch.Tensor:
    """Validate and convert a LayerSelection to the shared backend mask."""
    validate_selection(
        sel,
        expected_kind=expected_kind,
        n_prompt=n_prompt,
        budget=budget,
        sink=sink,
        recent=recent,
        H_q=H_q,
        group_size=group_size,
    )
    if sel.kind == "block":
        return build_allow_mask_keys(
            sel.blocks, n_prompt, sink, recent, device
        )
    return build_per_head_allow_mask(
        sel.per_head_idx,
        sel.per_head_valid,
        n_prompt,
        sink,
        recent,
        H_q,
        group_size,
        device,
    )
