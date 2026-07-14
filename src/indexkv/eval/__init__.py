"""Evaluation loaders, prompts, and official scorers.

Scope: LongBench v2 (multiple-choice, official `result.py` scoring) and RULER
(string-match, official NVIDIA / kvpress scoring). LongBench v1 is intentionally
out of scope for this harness.
"""

from __future__ import annotations

import importlib


_LONGBENCH2_EXPORTS = frozenset({
    "LEADERBOARD_TARGETS",
    "LongBench2Score",
    "build_prompt",
    "build_cot_answer_prompt",
    "compare_to_leaderboard",
    "extract_answer",
    "iter_official_items",
    "load_longbench2",
    "official_item",
    "score_predictions",
    "truncate_prompt",
})
_RULER_EXPORTS = frozenset({
    "RULER_TASKS",
    "load_ruler",
    "max_new_tokens",
    "ruler_sample_score",
    "ruler_task_score",
    "string_match_all",
    "string_match_part",
    "task_category",
})

__all__ = [
    # LongBench v2
    "LEADERBOARD_TARGETS",
    "LongBench2Score",
    "build_prompt",
    "build_cot_answer_prompt",
    "compare_to_leaderboard",
    "extract_answer",
    "iter_official_items",
    "load_longbench2",
    "official_item",
    "score_predictions",
    "truncate_prompt",
    # RULER
    "RULER_TASKS",
    "load_ruler",
    "max_new_tokens",
    "ruler_sample_score",
    "ruler_task_score",
    "string_match_all",
    "string_match_part",
    "task_category",
]


def __getattr__(name: str):
    """Load only the benchmark adapter that owns the requested symbol."""
    if name in _LONGBENCH2_EXPORTS:
        module_name = ".longbench2"
    elif name in _RULER_EXPORTS:
        module_name = ".ruler"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value
