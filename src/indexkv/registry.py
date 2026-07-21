"""Registry for the index-only method comparison.

Each method builds a budget-independent per-layer index once and answers one or
more budgeted selection queries from that index. The registry only tracks what
the engine needs to run a method fairly (selection shape, reselection policy,
prefill inputs, dense-reference scope) plus a short human-readable reference.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Type

from .types import (
    LayerSelection,
    MethodConfig,
    MethodScope,
    QueryNeeds,
    ReselectPolicy,
    SelectionKind,
)


class IndexMethod:
    """Base class for an index-only sparse-attention method.

    Lifecycle per layer and sample:

    ``index = build(K, V, cfg_with_budget_zero, Q=prefill_query_or_none)``
    ``selection = select(index, live_query_or_none, cfg_with_budget)``

    Static methods consume their declared prefill-Q slice during ``build`` and
    store budget-independent ranking state in the returned index.  Per-step
    methods consume the live post-RoPE decode query during ``select``.
    """

    name: str = ""
    kind: SelectionKind = "block"
    needs: QueryNeeds = QueryNeeds()
    reselect: ReselectPolicy = "per_step"
    scope: MethodScope = "index_only"
    implemented: bool = False
    reference: str = ""

    def build(self, K, V, cfg: MethodConfig, Q=None):
        raise NotImplementedError

    def select(self, index, Q, cfg: MethodConfig) -> LayerSelection:
        raise NotImplementedError

    def has_kv_transform(self, index, cfg: MethodConfig) -> bool:
        """Return whether the optional gathered-KV hook is active."""
        return False

    def transform_selected_kv(
        self,
        index,
        k,
        v,
        positions,
        cfg: MethodConfig,
    ):
        """Optional numerical KV transform after sparse gather.

        Index methods are exact-KV by default.  Adapters may override this for
        an explicitly requested source-fidelity codec emulation; the backend
        still retains the exact cache, so this hook must not be used to claim
        packed-cache memory or kernel speedups.
        """
        return k, v


METHODS: Dict[str, Type[IndexMethod]] = {}


def register(
    name: str,
    *,
    kind: SelectionKind = "block",
    needs: Optional[QueryNeeds] = None,
    reselect: ReselectPolicy = "per_step",
    scope: MethodScope = "index_only",
    implemented: bool = True,
    reference: str = "",
) -> Callable[[Type[IndexMethod]], Type[IndexMethod]]:
    """Register one comparison method and the metadata the engine needs."""

    def deco(cls: Type[IndexMethod]) -> Type[IndexMethod]:
        if name in METHODS:
            raise ValueError(f"method '{name}' already registered")
        n = needs or QueryNeeds()
        if reselect == "per_step" and n.query != "last":
            raise ValueError(
                f"method '{name}': per_step methods receive the live decode "
                f"query and must declare needs.query='last', got '{n.query}'"
            )
        cls.name = name
        cls.kind = kind
        cls.needs = n
        cls.reselect = reselect
        cls.scope = scope
        cls.implemented = implemented
        cls.reference = reference
        METHODS[name] = cls
        return cls

    return deco


def get_method(name: str) -> IndexMethod:
    if name not in METHODS:
        raise KeyError(
            f"unknown method '{name}'. Registered: {sorted(METHODS)}"
        )
    return METHODS[name]()


def list_methods() -> Dict[str, Dict[str, object]]:
    """Return method metadata for CLI listings and experiment manifests."""
    return {
        n: {
            "kind": c.kind,
            "reselect": c.reselect,
            "scope": c.scope,
            "implemented": c.implemented,
            "reference": c.reference,
        }
        for n, c in sorted(METHODS.items())
    }
