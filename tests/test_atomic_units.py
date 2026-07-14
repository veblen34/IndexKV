"""Deterministic tests for the atomic-retrieval-unit fairness rule.

Every method covered here must return a union of complete units from the
budget-independent index. A unit that does not fit is skipped rather than
truncated, so a short tail unit may still be selected when ``B`` is smaller
than the method's nominal unit size.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indexkv.methods.chunkkv import ChunkKV
from indexkv.methods.selfindexing import SelfIndexing
from indexkv.methods.wave_index import WaveIndex, WaveIndexData
from indexkv.types import LayerSelection, MethodConfig


N = 10
UNIT = 4
TAIL = {8, 9}


def make_cfg(budget: int, **extra) -> MethodConfig:
    return MethodConfig(
        budget=budget,
        block_size=UNIT,
        sink=0,
        recent=0,
        group_size=1,
        n_prompt=N,
        device="cpu",
        dtype=torch.float32,
        extra=extra,
    )


def selected_ids(selection: LayerSelection, head: int = 0) -> set[int]:
    indices = selection.per_head_idx[head]
    valid = selection.per_head_valid
    if valid is not None:
        indices = indices[valid[head]]
    return {int(position) for position in indices.tolist()}


class AtomicUnitTests(unittest.TestCase):
    def assert_units_partition_prompt(self, units: list[set[int]]) -> None:
        covered: set[int] = set()
        for unit in units:
            self.assertTrue(unit)
            self.assertTrue(covered.isdisjoint(unit))
            covered.update(unit)
        self.assertEqual(covered, set(range(N)))

    def assert_block_selection_is_atomic(
        self,
        selection: LayerSelection,
        indexed_blocks: torch.Tensor,
        budget: int,
    ) -> None:
        self.assertEqual(selection.kind, "block")
        original = [
            (int(start), int(end))
            for start, end in indexed_blocks.tolist()
        ]
        chosen = [
            (int(start), int(end))
            for start, end in selection.blocks.tolist()
        ]
        self.assertEqual(len(chosen), len(set(chosen)))
        self.assertTrue(all(block in original for block in chosen))

        units = [set(range(start, end)) for start, end in original]
        self.assert_units_partition_prompt(units)
        selected = {
            position
            for start, end in chosen
            for position in range(start, end)
        }
        self.assertLessEqual(len(selected), budget)
        complete_union: set[int] = set()
        for unit in units:
            overlap = selected & unit
            self.assertTrue(not overlap or overlap == unit)
            if overlap:
                complete_union.update(unit)
        self.assertEqual(selected, complete_union)

    def assert_per_head_selection_is_atomic(
        self,
        selection: LayerSelection,
        units_by_head: list[list[set[int]]],
        budget: int,
    ) -> None:
        self.assertEqual(selection.kind, "per_head")
        for head, units in enumerate(units_by_head):
            self.assert_units_partition_prompt(units)
            indices = selection.per_head_idx[head]
            valid = selection.per_head_valid
            if valid is not None:
                indices = indices[valid[head]]
            values = [int(position) for position in indices.tolist()]
            self.assertEqual(len(values), len(set(values)))
            self.assertLessEqual(len(values), budget)

            selected = set(values)
            complete_union: set[int] = set()
            for unit in units:
                overlap = selected & unit
                self.assertTrue(not overlap or overlap == unit)
                if overlap:
                    complete_union.update(unit)
            self.assertEqual(selected, complete_union)

    def test_chunkkv_greedy_packs_complete_tail_chunk(self) -> None:
        method = ChunkKV()
        keys = torch.zeros(1, N, 1)
        prefill_query = torch.zeros(1, N, 1)
        index = method.build(
            keys,
            keys,
            make_cfg(0, chunk_length=UNIT, kernel_size=1),
            Q=prefill_query,
        )
        self.assertEqual(index.blocks.tolist(), [[0, 4], [4, 8], [8, 10]])

        for budget in (0, 2, 3, 6):
            with self.subTest(budget=budget):
                selection = method.select(
                    index,
                    None,
                    make_cfg(budget, chunk_length=UNIT, kernel_size=1),
                )
                self.assert_block_selection_is_atomic(
                    selection, index.blocks, budget
                )
        self.assertEqual(
            method.select(
                index,
                None,
                make_cfg(0, chunk_length=UNIT, kernel_size=1),
            ).blocks.numel(),
            0,
        )
        tail_only = method.select(
            index,
            None,
            make_cfg(2, chunk_length=UNIT, kernel_size=1),
        )
        self.assertEqual(tail_only.blocks.tolist(), [[8, 10]])

    def test_selfindexing_vq_lut_selects_per_token(self) -> None:
        # Faithful selfindexing is the repo's VQ/LUT selector: sign-orthant product
        # quantization (head_dim split into SDIM=4-dim subspaces -> 16 codes each)
        # scored by a LUT approximation of q.k, top-`budget` TOKENS per kv-head.
        method = SelfIndexing()
        torch.manual_seed(0)
        keys = torch.randn(1, N, 8)                     # head_dim 8 -> SUB=2 subspaces
        index = method.build(keys, keys, make_cfg(0))
        self.assertEqual(tuple(index.codes.shape), (1, N, 2))      # (H, N, SUB)
        self.assertEqual(tuple(index.codebook.shape), (1, 2, 16, 4))  # (H, SUB, 16, SDIM)
        query = torch.randn(1, 1, 8)

        # budget 0 -> empty; budget k -> exactly k unique per-token positions
        self.assertEqual(selected_ids(method.select(index, query, make_cfg(0))), set())
        for budget in (1, 3, 6):
            with self.subTest(budget=budget):
                ids = selected_ids(method.select(index, query, make_cfg(budget)))
                self.assertEqual(len(ids), budget)                  # per-token, exactly budget
                self.assertTrue(ids.issubset(set(range(N))))

        # selection matches exact top-k of the LUT approx-q.k score (not chunk-atomic)
        sel = method.select(index, query, make_cfg(5))
        qsub = query[0, -1].reshape(1, 2, 4)
        table = torch.einsum("hscd,hsd->hsc", index.codebook, qsub)
        scores = torch.gather(table, 2, index.codes.permute(0, 2, 1)).sum(1)  # (H, N)
        self.assertEqual(
            selected_ids(sel), set(scores[0].topk(5).indices.tolist())
        )

    def test_wave_index_returns_unbroken_index_clusters(self) -> None:
        method = WaveIndex()
        # An explicit deterministic CSR index isolates the retrieval rule from
        # k-means initialization: the final cluster is the short tail unit.
        index = WaveIndexData(
            centroids=torch.tensor([[[1.0], [2.0], [9.0]]]),
            members=torch.arange(N, dtype=torch.long).view(1, N),
            offsets=torch.tensor([[0, 4, 8, 10]], dtype=torch.long),
            sizes=torch.tensor([[4, 4, 2]], dtype=torch.long),
            lo=0,
            num_segments=1,
        )
        units = [
            set(
                int(position)
                for position in index.members[
                    0, index.offsets[0, cluster]:index.offsets[0, cluster + 1]
                ].tolist()
            )
            for cluster in range(index.sizes.shape[1])
        ]
        self.assertEqual([len(unit) for unit in units], [4, 4, 2])
        query = torch.ones(1, 1, 1)

        for budget in (0, 2, 3, 6):
            with self.subTest(budget=budget):
                selection = method.select(index, query, make_cfg(budget))
                self.assert_per_head_selection_is_atomic(
                    selection, [units], budget
                )
        self.assertEqual(
            selected_ids(method.select(index, query, make_cfg(0))),
            set(),
        )
        self.assertEqual(
            selected_ids(method.select(index, query, make_cfg(2))),
            TAIL,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
