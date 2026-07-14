"""Runner configuration and trained-index shape validation tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import indexkv.methods  # noqa: F401
from indexkv.backends.base import ModelDims
from indexkv.runner import SweepRunner, _partition_decode_timing, parse_set_overrides


class FourLayerBackend:
    dims = ModelDims(
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
    )

    def sparse_generate(self, prompt_ids, max_new, provider):
        return "ok"


class RunnerConfigTests(unittest.TestCase):
    def test_decode_timing_partition_is_scope_aware(self):
        exclusive, error = _partition_decode_timing(2.0, 0.25, "decode")
        self.assertEqual(exclusive, 1.75)
        self.assertEqual(error, 0.0)

        for scope in ("provider_setup", "none"):
            query = 0.25 if scope == "provider_setup" else 0.0
            with self.subTest(scope=scope):
                exclusive, error = _partition_decode_timing(2.0, query, scope)
                self.assertEqual(exclusive, 2.0)
                self.assertEqual(error, 0.0)

        # Early EOS can produce no sparse selection calls.
        exclusive, error = _partition_decode_timing(0.1, 0.0, "decode")
        self.assertEqual(exclusive, 0.1)
        self.assertEqual(error, 0.0)

        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            _partition_decode_timing(0.1, 0.2, "decode")
        with self.assertRaisesRegex(RuntimeError, "nonzero"):
            _partition_decode_timing(0.1, 0.01, "none")

    def test_parse_set_values_are_scoped_and_typed(self):
        parsed = parse_set_overrides(
            [
                "selfindexing.m=4",
                "selfindexing.enabled=true",
                "selfindexing.optional=null",
                "chunkkv.ratio=1.25",
                "wave_index.label=demo",
            ]
        )
        self.assertEqual(
            parsed,
            {
                "selfindexing": {
                    "m": 4,
                    "enabled": True,
                    "optional": None,
                },
                "chunkkv": {"ratio": 1.25},
                "wave_index": {"label": "demo"},
            },
        )

    def test_parse_set_rejects_malformed_and_unknown_targets(self):
        malformed = [
            "chunkkv.m",
            "m=1",
            ".m=1",
            "chunkkv.=1",
            "not_registered.knob=1",
        ]
        for value in malformed:
            with self.subTest(value=value):
                with self.assertRaises(SystemExit):
                    parse_set_overrides([value])

    def test_runner_validates_common_configuration(self):
        backend = FourLayerBackend()
        runner = SweepRunner(
            backend,
            ["full"],
            model_name="fake",
            device="cpu",
            dtype=torch.float32,
        )
        self.assertEqual(runner.dense_prefix_layers, 2)
        self.assertEqual(runner.runnable, ["full"])

        with self.assertRaises(ValueError):
            SweepRunner(
                backend,
                ["full"],
                model_name="fake",
                block_size=0,
                device="cpu",
            )
        with self.assertRaises(ValueError):
            SweepRunner(
                backend,
                ["full"],
                model_name="fake",
                sink=-1,
                device="cpu",
            )
        with self.assertRaises(ValueError):
            SweepRunner(
                backend,
                ["full"],
                model_name="fake",
                dense_prefix_layers=5,
                device="cpu",
            )
        with self.assertRaisesRegex(ValueError, "outside this sweep"):
            SweepRunner(
                backend,
                ["full"],
                model_name="fake",
                overrides={"chunkkv": {"m": 2}},
                device="cpu",
            )

    def test_runner_rejects_duplicate_method_names(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            SweepRunner(
                FourLayerBackend(),
                ["full", "full"],
                model_name="fake",
                device="cpu",
            )

    def test_sparse_generate_requires_budget(self):
        class MinimalBackend(FourLayerBackend):
            pass

        runner = SweepRunner(
            MinimalBackend(),
            ["range_search"],
            model_name="fake",
            dense_prefix_layers=2,
            device="cpu",
            dtype=torch.float32,
            overrides={"range_search": {"m": 2}},
        )
        with self.assertRaisesRegex(ValueError, "requires a budget"):
            runner.generate(
                "range_search",
                indices={},
                cap=None,
                budget=None,
                ids=torch.zeros(1, 1, dtype=torch.long),
                max_new=1,
            )


if __name__ == "__main__":
    unittest.main()
