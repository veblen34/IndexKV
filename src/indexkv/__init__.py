"""Unified index-only evaluation framework for KV-cache index-selection baselines.

Public surface:
    from indexkv import METHODS, get_method, list_methods
    from indexkv.engine import DenseProvider, build_indices, make_provider
    from indexkv.backends.llama import LlamaBackend
"""

from __future__ import annotations

from .registry import METHODS, IndexMethod, get_method, list_methods, register
from .types import LayerSelection, MethodConfig, MethodRole, MethodScope, QueryNeeds

# Importing the methods package triggers registration of every method.
from . import methods  # noqa: F401,E402

__all__ = [
    "METHODS",
    "IndexMethod",
    "get_method",
    "list_methods",
    "register",
    "LayerSelection",
    "MethodConfig",
    "MethodRole",
    "MethodScope",
    "QueryNeeds",
]
