"""Fairness-contract tests with no model or dataset dependency."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import indexkv.methods  # noqa: F401
from indexkv.backends.base import Capture, ModelDims
from indexkv.engine import DenseProvider, build_indices, make_provider
from indexkv.masks import selection_to_mask, validate_selection
from indexkv.registry import METHODS, IndexMethod, get_method
from indexkv.types import LayerSelection, MethodConfig, QueryNeeds


H_KV = 2
GROUP_SIZE = 2
H_Q = H_KV * GROUP_SIZE
HEAD_DIM = 8
N_PROMPT = 80
LAYERS = (0, 1, 2)
DENSE_PREFIX = 2
SINK = 4
RECENT = 8


class FakeBackend:
    dims = ModelDims(
        n_layers=len(LAYERS),
        n_heads=H_Q,
        n_kv_heads=H_KV,
        head_dim=HEAD_DIM,
    )


def make_capture(seed: int = 0) -> Capture:
    generator = torch.Generator().manual_seed(seed)
    return Capture(
        K={
            layer: torch.randn(
                H_KV, N_PROMPT, HEAD_DIM, generator=generator
            )
            for layer in LAYERS
        },
        Q={
            layer: torch.randn(
                H_Q, N_PROMPT, HEAD_DIM, generator=generator
            )
            for layer in LAYERS
        },
        V={
            layer: torch.randn(
                H_KV, N_PROMPT, HEAD_DIM, generator=generator
            )
            for layer in LAYERS
        },
        n_prompt=N_PROMPT,
    )


def engine_kwargs() -> dict:
    return {
        "block_size": 16,
        "sink": SINK,
        "recent": RECENT,
        "dense_prefix_layers": DENSE_PREFIX,
        "device": "cpu",
        "dtype": torch.float32,
    }


class CountingPerStep(IndexMethod):
    name = "counting_per_step"
    kind = "per_head"
    needs = QueryNeeds(query="last")
    reselect = "per_step"
    scope = "index_only"

    def __init__(self):
        self.build_calls = []
        self.select_queries = []

    def build(self, K, V, cfg: MethodConfig, Q=None):
        self.build_calls.append((cfg.layer_idx, cfg.budget, Q))
        return {"layer": cfg.layer_idx}

    def select(self, index, Q, cfg: MethodConfig) -> LayerSelection:
        self.select_queries.append(Q.clone())
        position = SINK if float(Q.sum()) < 0 else SINK + 1
        return LayerSelection(
            kind="per_head",
            per_head_idx=torch.tensor([[position]], dtype=torch.long),
        )


class CountingStatic(IndexMethod):
    name = "counting_static"
    kind = "block"
    needs = QueryNeeds(query="obs_window", obs_window=3)
    reselect = "static"
    scope = "index_only"

    def __init__(self):
        self.build_calls = []
        self.select_queries = []

    def build(self, K, V, cfg: MethodConfig, Q=None):
        if Q is None:
            raise AssertionError("static build did not receive prefill Q")
        self.build_calls.append((cfg.layer_idx, cfg.budget, Q.clone()))
        return {"layer": cfg.layer_idx}

    def select(self, index, Q, cfg: MethodConfig) -> LayerSelection:
        self.select_queries.append(Q)
        if cfg.budget == 0:
            blocks = torch.empty(0, 2, dtype=torch.long)
        else:
            width = min(cfg.budget, 2)
            blocks = torch.tensor(
                [[SINK, SINK + width]], dtype=torch.long
            )
        return LayerSelection(kind="block", blocks=blocks)


class SelectionContractTests(unittest.TestCase):
    def validate(self, selection, *, kind="per_head", budget=4):
        return validate_selection(
            selection,
            expected_kind=kind,
            n_prompt=12,
            budget=budget,
            sink=2,
            recent=2,
            H_q=4,
            group_size=2,
        )

    def test_layer_selection_payload_and_shape_are_strict(self):
        blocks = torch.empty(0, 2, dtype=torch.long)
        indices = torch.empty(2, 0, dtype=torch.long)
        with self.assertRaises(ValueError):
            LayerSelection(
                kind="block", blocks=blocks, per_head_idx=indices
            )
        with self.assertRaises(ValueError):
            LayerSelection(kind="per_head", blocks=blocks)
        with self.assertRaises(TypeError):
            LayerSelection(
                kind="block", blocks=torch.empty(0, 2)
            )
        with self.assertRaises(ValueError):
            LayerSelection(
                kind="per_head",
                per_head_idx=torch.empty(2, 0, 1, dtype=torch.long),
            )

    def test_invalid_heads_forced_positions_duplicates_and_budget_fail(self):
        bad_heads = LayerSelection(
            kind="per_head",
            per_head_idx=torch.tensor(
                [[2], [3], [4]], dtype=torch.long
            ),
        )
        with self.assertRaisesRegex(ValueError, "rows"):
            self.validate(bad_heads)

        for forced_position in (1, 10):
            with self.subTest(forced_position=forced_position):
                selection = LayerSelection(
                    kind="per_head",
                    per_head_idx=torch.tensor(
                        [[forced_position]], dtype=torch.long
                    ),
                )
                with self.assertRaisesRegex(ValueError, "eligible"):
                    self.validate(selection)

        duplicate = LayerSelection(
            kind="per_head",
            per_head_idx=torch.tensor([[3, 3]], dtype=torch.long),
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.validate(duplicate)

        over_budget = LayerSelection(
            kind="per_head",
            per_head_idx=torch.tensor([[3, 4]], dtype=torch.long),
        )
        with self.assertRaisesRegex(ValueError, "budget"):
            self.validate(over_budget, budget=1)

        overlapping_blocks = LayerSelection(
            kind="block",
            blocks=torch.tensor([[2, 5], [4, 6]], dtype=torch.long),
        )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.validate(
                overlapping_blocks, kind="block", budget=8
            )

        forced_block = LayerSelection(
            kind="block",
            blocks=torch.tensor([[1, 3]], dtype=torch.long),
        )
        with self.assertRaisesRegex(ValueError, "eligible"):
            self.validate(forced_block, kind="block", budget=4)

    def test_ragged_sentinel_and_kv_head_broadcast(self):
        selection = LayerSelection(
            kind="per_head",
            per_head_idx=torch.tensor(
                [[4, -1, 7], [5, 6, -1]], dtype=torch.long
            ),
            per_head_valid=torch.tensor(
                [[True, False, True], [True, True, False]]
            ),
        )
        self.validate(selection, budget=2)
        mask = selection_to_mask(
            selection,
            expected_kind="per_head",
            n_prompt=12,
            budget=2,
            sink=2,
            recent=2,
            H_q=4,
            group_size=2,
            device="cpu",
        ).reshape(4, 12)
        self.assertTrue(torch.equal(mask[0], mask[1]))
        self.assertTrue(torch.equal(mask[2], mask[3]))
        self.assertTrue(mask[0, [4, 7]].all())
        self.assertTrue(mask[2, [5, 6]].all())
        self.assertFalse(bool(mask[0, 5]))
        self.assertFalse(bool(mask[2, 4]))

        invalid_padding = LayerSelection(
            kind="per_head",
            per_head_idx=torch.tensor([[4, 0]], dtype=torch.long),
            per_head_valid=torch.tensor([[True, False]]),
        )
        with self.assertRaisesRegex(ValueError, "sentinel"):
            self.validate(invalid_padding)

    def test_budget_zero_keeps_only_common_windows(self):
        selection = LayerSelection(
            kind="per_head",
            per_head_idx=torch.empty(2, 0, dtype=torch.long),
        )
        self.validate(selection, budget=0)
        mask = selection_to_mask(
            selection,
            expected_kind="per_head",
            n_prompt=12,
            budget=0,
            sink=2,
            recent=2,
            H_q=4,
            group_size=2,
            device="cpu",
        ).reshape(4, 12)
        expected = torch.zeros(12, dtype=torch.bool)
        expected[:2] = True
        expected[-2:] = True
        self.assertTrue(torch.equal(mask, expected.expand(4, 12)))

    def test_shared_single_row_selection_broadcasts_to_all_heads(self):
        selection = LayerSelection(
            kind="per_head",
            per_head_idx=torch.tensor([[3, 8]], dtype=torch.long),
        )
        mask = selection_to_mask(
            selection,
            expected_kind="per_head",
            n_prompt=12,
            budget=2,
            sink=2,
            recent=2,
            H_q=4,
            group_size=2,
            device="cpu",
        ).reshape(4, 12)
        for head in range(1, 4):
            self.assertTrue(torch.equal(mask[0], mask[head]))


class EngineLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.capture = make_capture(13)
        self.q_negative = -torch.ones(H_Q, 1, HEAD_DIM)
        self.q_positive = torch.ones(H_Q, 1, HEAD_DIM)

    def test_build_once_reused_across_budgets_and_per_step_reselects(self):
        method = CountingPerStep()
        indices = build_indices(
            self.backend, method, self.capture, **engine_kwargs()
        )
        self.assertIsNone(indices[0])
        self.assertIsNone(indices[1])
        self.assertEqual(len(method.build_calls), 1)
        self.assertEqual(method.build_calls[0][:2], (2, 0))
        self.assertIsNone(method.build_calls[0][2])

        provider_one = make_provider(
            self.backend,
            method,
            self.capture,
            indices,
            1,
            **engine_kwargs(),
        )
        provider_two = make_provider(
            self.backend,
            method,
            self.capture,
            indices,
            2,
            **engine_kwargs(),
        )
        self.assertEqual(len(method.build_calls), 1)
        self.assertIsNone(provider_one.mask_for(0, self.q_negative))
        self.assertIsNone(provider_one.mask_for(1, self.q_negative))

        first = provider_one.mask_for(2, self.q_negative)
        second = provider_one.mask_for(2, self.q_positive)
        self.assertEqual(len(method.select_queries), 2)
        self.assertFalse(torch.equal(first, second))
        stats = provider_one.stats()
        self.assertEqual(stats["selection_calls"], {2: 2})
        self.assertEqual(stats["selection_aggregate"][2]["max"], 1)
        self.assertGreaterEqual(stats["selection_s_total"], 0.0)
        self.assertEqual(stats["selected_counts"], {})
        self.assertEqual(len(method.build_calls), 1)
        self.assertEqual(provider_two.indices, indices)

    def test_static_build_consumes_prefill_q_and_select_gets_none(self):
        method = CountingStatic()
        indices = build_indices(
            self.backend, method, self.capture, **engine_kwargs()
        )
        self.assertEqual(len(method.build_calls), 1)
        layer, build_budget, captured_q = method.build_calls[0]
        self.assertEqual((layer, build_budget), (2, 0))
        self.assertTrue(
            torch.equal(captured_q, self.capture.Q[2][:, -3:, :])
        )

        provider_one = make_provider(
            self.backend,
            method,
            self.capture,
            indices,
            1,
            **engine_kwargs(),
        )
        provider_two = make_provider(
            self.backend,
            method,
            self.capture,
            indices,
            2,
            **engine_kwargs(),
        )
        self.assertEqual(len(method.build_calls), 1)
        self.assertEqual(method.select_queries, [None, None])

        before = len(method.select_queries)
        first = provider_one.mask_for(2, self.q_negative)
        second = provider_one.mask_for(2, self.q_positive)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(len(method.select_queries), before)
        stats_one = provider_one.stats()
        stats_two = provider_two.stats()
        self.assertEqual(stats_one["selection_calls"], {2: 1})
        self.assertEqual(stats_two["selection_calls"], {2: 1})
        self.assertEqual(
            stats_one["selection_aggregate"][2]["max"], 1
        )
        self.assertEqual(stats_one["selection_aggregate"][2]["sample_count"], H_Q)
        self.assertEqual(
            stats_two["selection_aggregate"][2]["max"], 2
        )

    def test_dense_provider_returns_none_for_every_layer(self):
        provider = DenseProvider()
        for layer in LAYERS:
            self.assertIsNone(provider.mask_for(layer, self.q_positive))
        self.assertEqual(
            provider.stats(),
            {
                "selection_calls": {},
                "selection_aggregate": {},
                "selected_counts": {},
                "selection_s_total": 0.0,
            },
        )


class AllMethodsBudgetTests(unittest.TestCase):
    @staticmethod
    def method_weights(name: str, generator):
        if name == "hata":
            return {
                2: torch.randn(
                    H_KV, HEAD_DIM, 16, generator=generator
                )
            }
        if name == "hashattention":
            def mlp():
                dimensions = (12, 12, 8)
                current_input = HEAD_DIM
                layers = []
                for output in dimensions:
                    layers.append(
                        (
                            torch.randn(
                                H_Q,
                                output,
                                current_input,
                                generator=generator,
                            ),
                            torch.randn(
                                H_Q, output, generator=generator
                            ),
                        )
                    )
                    current_input = output
                return layers

            return {2: {"k": mlp(), "q": mlp()}}
        return None

    @staticmethod
    def method_extra(name: str):
        return {
            "chunkkv": {"chunk_length": 10, "kernel_size": 5},
            "selfindexing": {"chunk_size": 8},
            "wave_index": {
                "avg_cluster_size": 8,
                "kmeans_iter": 2,
                "assignment_chunk_size": 16,
            },
        }.get(name)

    def assert_mask_contract(self, mask, budget):
        self.assertEqual(mask.dtype, torch.bool)
        self.assertIn(
            tuple(mask.shape),
            ((1, 1, 1, N_PROMPT), (1, H_Q, 1, N_PROMPT)),
        )
        flat = mask.reshape(-1, N_PROMPT)
        self.assertTrue(flat[:, :SINK].all())
        self.assertTrue(flat[:, N_PROMPT - RECENT:].all())
        middle_counts = flat[:, SINK:N_PROMPT - RECENT].sum(dim=-1)
        self.assertTrue(bool((middle_counts <= budget).all()))
        if budget == 0:
            self.assertTrue(bool((middle_counts == 0).all()))

    def test_all_index_methods_respect_zero_and_small_budgets(self):
        backend = FakeBackend()
        capture = make_capture(21)
        query = torch.randn(
            H_Q, 1, HEAD_DIM, generator=torch.Generator().manual_seed(22)
        )
        generator = torch.Generator().manual_seed(23)
        method_names = [
            name
            for name, method_class in sorted(METHODS.items())
            if method_class.implemented
            and method_class.scope == "index_only"
        ]
        self.assertEqual(
            method_names,
            [
                "chunkkv",
                "hashattention",
                "hata",
                "range_search",
                "selfindexing",
                "wave_index",
            ],
        )

        for name in method_names:
            with self.subTest(method=name):
                method = get_method(name)
                weights = self.method_weights(name, generator)
                extra = self.method_extra(name)
                indices = build_indices(
                    backend,
                    method,
                    capture,
                    weights=weights,
                    extra=extra,
                    **engine_kwargs(),
                )
                self.assertIsNone(indices[0])
                self.assertIsNone(indices[1])
                self.assertIsNotNone(indices[2])

                for budget in (0, 7):
                    provider = make_provider(
                        backend,
                        method,
                        capture,
                        indices,
                        budget,
                        weights=weights,
                        extra=extra,
                        **engine_kwargs(),
                    )
                    self.assertIsNone(provider.mask_for(0, query))
                    self.assertIsNone(provider.mask_for(1, query))
                    mask = provider.mask_for(2, query)
                    self.assertIsNotNone(mask)
                    self.assert_mask_contract(mask, budget)
                    stats = provider.stats()
                    self.assertEqual(
                        stats["selection_calls"].get(2), 1
                    )
                    aggregate = stats["selection_aggregate"][2]
                    self.assertGreater(aggregate["sample_count"], 0)
                    self.assertLessEqual(aggregate["max"], budget)
                    self.assertGreaterEqual(aggregate["min"], 0)
                    self.assertEqual(stats["selected_counts"], {})


if __name__ == "__main__":
    unittest.main()
