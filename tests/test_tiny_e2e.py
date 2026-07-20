"""Tiny local Llama end-to-end test for the shared sparse backend."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from transformers import LlamaConfig, LlamaForCausalLM
except ModuleNotFoundError:
    LlamaConfig = None
    LlamaForCausalLM = None

if LlamaConfig is not None:
    from indexkv.backends.llama import LlamaBackend
    from indexkv.engine import DenseProvider
    from indexkv.runner import SweepRunner


H_KV = 2
H_Q = 4
D_HEAD = 32
LAYERS = 4
DENSE_PREFIX = 2
VOCAB_SIZE = 128


class LocalTokenizer:
    """Minimal tokenizer surface consumed by LlamaBackend."""

    pad_token_id = 0
    bos_token_id = 1
    # Deliberately outside the model vocabulary so random decoding always
    # exercises at least one provider-controlled decode step.
    eos_token_id = VOCAB_SIZE + 1

    def __len__(self):
        return VOCAB_SIZE

    def _encode(self, text: str) -> torch.Tensor:
        tokens = [self.bos_token_id]
        tokens.extend(3 + (ord(char) % (VOCAB_SIZE - 3)) for char in text)
        return torch.tensor([tokens], dtype=torch.long)

    def __call__(
        self, text, *, return_tensors="pt", add_special_tokens=True
    ):
        if return_tensors != "pt":
            raise ValueError("LocalTokenizer only supports return_tensors='pt'")
        return SimpleNamespace(input_ids=self._encode(text))

    def apply_chat_template(
        self, messages, *, add_generation_prompt=True, return_tensors="pt"
    ):
        text = "\n".join(message["content"] for message in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return self._encode(text)

    def decode(self, token_ids, *, skip_special_tokens=True):
        values = torch.as_tensor(token_ids).reshape(-1).tolist()
        if skip_special_tokens:
            values = [
                value
                for value in values
                if value not in (self.pad_token_id, self.bos_token_id)
            ]
        return " ".join(str(value) for value in values)


@unittest.skipUnless(
    LlamaConfig is not None,
    "transformers is not installed; skipping local tiny Llama E2E",
)
class TinyLlamaE2ETests(unittest.TestCase):
    def tiny_backend(self):
        tokenizer = LocalTokenizer()
        config = LlamaConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=H_Q * D_HEAD,
            intermediate_size=96,
            num_hidden_layers=LAYERS,
            num_attention_heads=H_Q,
            num_key_value_heads=H_KV,
            max_position_embeddings=512,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=2,
            pad_token_id=tokenizer.pad_token_id,
        )
        torch.manual_seed(0)
        model = LlamaForCausalLM(config).eval().to(torch.float32)
        return LlamaBackend(model, tokenizer)

    @staticmethod
    def fake_weights_on_disk(root: Path):
        hata_dir = root / "hata_tiny"
        hata_dir.mkdir()
        generator = torch.Generator().manual_seed(1)
        for layer in range(DENSE_PREFIX, LAYERS):
            weight = torch.randn(
                H_KV, D_HEAD, 32, generator=generator
            )
            torch.save(
                weight,
                hata_dir / f"hash_weight_layer_{layer:02d}.pt",
            )

        state_dict = {}
        linear_shapes = (
            (0, 24, D_HEAD),
            (2, 24, 24),
            (4, 8, 24),
        )
        for layer in range(DENSE_PREFIX, LAYERS):
            for key_or_query in ("k", "q"):
                for head in range(H_Q):
                    prefix = (
                        f"{layer}.learning_to_hash_transformation_"
                        f"{key_or_query}.{head}"
                    )
                    for sequential_index, output_dim, input_dim in linear_shapes:
                        state_dict[
                            f"{prefix}.{sequential_index}.weight"
                        ] = torch.randn(
                            output_dim, input_dim, generator=generator
                        )
                        state_dict[
                            f"{prefix}.{sequential_index}.bias"
                        ] = torch.randn(output_dim, generator=generator)

        hash_file = root / "hashattention_tiny.pt"
        torch.save(state_dict, hash_file)
        return hata_dir, hash_file

    def test_all_methods_and_dense_provider_use_real_backend_path(self):
        backend = self.tiny_backend()
        temp_root = None
        with tempfile.TemporaryDirectory(prefix="indexkv_tiny_") as tmp:
            temp_root = Path(tmp)
            hata_dir, hash_file = self.fake_weights_on_disk(temp_root)
            methods = [
                "full",
                "hata",
                "hashattention",
                "wave_index",
                "range_search",
                "selfindexing",
                "chunkkv",
            ]
            runner = SweepRunner(
                backend,
                methods,
                model_name="tiny",
                block_size=16,
                sink=4,
                recent=16,
                dense_prefix_layers=DENSE_PREFIX,
                device="cpu",
                dtype=torch.float32,
                overrides={
                    "hata": {
                        "weights_path": str(hata_dir),
                        "rbits": 32,
                    },
                    "hashattention": {
                        "weights_path": str(hash_file)
                    },
                    "wave_index": {
                        "avg_cluster_size": 8,
                        "kmeans_iter": 2,
                        "assignment_chunk_size": 16,
                    },
                    "selfindexing": {
                        "chunk_size": 8,
                        "emulate_2bit_kv": True,
                    },
                    "chunkkv": {
                        "chunk_length": 8,
                        "kernel_size": 5,
                    },
                },
            )
            self.assertFalse(runner.skipped, runner.skipped)

            prompt = "index only fairness benchmark " * 4
            ids = backend.tokenize(prompt, chat_template=False)
            prompt_length = ids.shape[1]
            self.assertGreater(prompt_length, 64)

            capture = runner.capture(ids)
            self.assertEqual(capture.n_prompt, prompt_length)
            self.assertEqual(set(capture.K), set(range(LAYERS)))
            expected_q = min(runner.q_window, prompt_length)
            self.assertEqual(
                capture.K[0].shape,
                (H_KV, prompt_length, D_HEAD),
            )
            self.assertEqual(
                capture.Q[0].shape,
                (H_Q, expected_q, D_HEAD),
            )

            dense = runner.generate(
                "full", None, capture, None, ids, max_new=2
            )
            for name in methods[1:]:
                indices, _ = runner.build(name, capture)
                self.assertIsNone(indices[0])
                self.assertIsNone(indices[1])
                self.assertIsNotNone(indices[2])
                self.assertIsNotNone(indices[3])
                generated = runner.generate(
                    name,
                    indices,
                    capture,
                    budget=32,
                    ids=ids,
                    max_new=2,
                )
                self.assertIsInstance(generated, str)

            class KeepAll:
                def mask_for(self, layer_idx, query):
                    return torch.ones(
                        1, 1, 1, prompt_length, dtype=torch.bool
                    )

            keep_all = backend.sparse_generate(
                ids, 2, KeepAll()
            )
            dense_provider = backend.sparse_generate(
                ids, 2, DenseProvider()
            )
            self.assertEqual(keep_all, dense_provider)
            self.assertEqual(dense, dense_provider)

            backend.prefill_mlp_chunk_size = 2
            chunked_capture = runner.capture(ids)
            for layer_idx in capture.K:
                torch.testing.assert_close(
                    chunked_capture.K[layer_idx], capture.K[layer_idx]
                )
                torch.testing.assert_close(
                    chunked_capture.Q[layer_idx], capture.Q[layer_idx]
                )
            chunked_dense = backend.sparse_generate(ids, 2, None)
            chunked_keep_all = backend.sparse_generate(
                ids, 2, KeepAll()
            )
            self.assertEqual(chunked_dense, dense)
            self.assertEqual(chunked_keep_all, dense_provider)
            backend.prefill_mlp_chunk_size = None

        self.assertIsNotNone(temp_root)
        self.assertFalse(temp_root.exists())


if __name__ == "__main__":
    require_dependencies = "--require-dependencies" in sys.argv
    if require_dependencies:
        sys.argv.remove("--require-dependencies")
    if LlamaConfig is None:
        print(
            "SKIP: transformers is not installed; "
            "tiny local Llama E2E was not run"
        )
        raise SystemExit(1 if require_dependencies else 0)
    unittest.main()
