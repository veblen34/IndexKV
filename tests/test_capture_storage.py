"""Backend capture invariants that run with only PyTorch installed.

Run directly with ``python tests/test_capture_storage.py``.  The tests install a
minimal Transformers module stub only when Transformers is unavailable; no model
weights or tokenizer files are required.
"""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import torch
from torch import nn


def _install_transformers_stub() -> None:
    try:
        import transformers  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    transformers = types.ModuleType("transformers")
    models = types.ModuleType("transformers.models")
    llama_pkg = types.ModuleType("transformers.models.llama")
    modeling = types.ModuleType("transformers.models.llama.modeling_llama")

    def identity_rope(q, k, cos, sin, *args, **kwargs):
        return q, k

    def repeat_kv(x, groups):
        return x.repeat_interleave(groups, dim=1)

    class LlamaAttention(nn.Module):
        def forward(self, hidden_states, past_key_value=None):  # pragma: no cover
            raise NotImplementedError

    modeling.apply_rotary_pos_emb = identity_rope
    modeling.repeat_kv = repeat_kv
    modeling.LlamaAttention = LlamaAttention
    llama_pkg.modeling_llama = modeling
    models.llama = llama_pkg
    transformers.models = models
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object

    sys.modules["transformers"] = transformers
    sys.modules["transformers.models"] = models
    sys.modules["transformers.models.llama"] = llama_pkg
    sys.modules["transformers.models.llama.modeling_llama"] = modeling


_install_transformers_stub()
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

# Keep this backend-only test independent of top-level method registration.
if "indexkv" not in sys.modules:
    indexkv_pkg = types.ModuleType("indexkv")
    indexkv_pkg.__path__ = [str(SRC_ROOT / "indexkv")]
    backends_pkg = types.ModuleType("indexkv.backends")
    backends_pkg.__path__ = [str(SRC_ROOT / "indexkv" / "backends")]
    sys.modules["indexkv"] = indexkv_pkg
    sys.modules["indexkv.backends"] = backends_pkg

from indexkv.backends.base import ModelBackend  # noqa: E402
from indexkv.backends import llama as llama_backend  # noqa: E402


class _FakeAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.v_proj = nn.Identity()
        self.width = width

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        return llama_backend.modeling_llama.apply_rotary_pos_emb(
            q, k, None, None
        )


class _FakeLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.self_attn = _FakeAttention(width)


class _FakeCaptureModel(nn.Module):
    def __init__(self, n_layers: int, width: int):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_FakeLayer(width) for _ in range(n_layers)]
        )


def _expect_runtime_error(contains: str, fn) -> None:
    try:
        fn()
    except RuntimeError as exc:
        assert contains in str(exc), (contains, str(exc))
    else:  # pragma: no cover - assertion helper
        raise AssertionError(f"expected RuntimeError containing {contains!r}")


def _expect_value_error(contains: str, fn) -> None:
    try:
        fn()
    except ValueError as exc:
        assert contains in str(exc), (contains, str(exc))
    else:  # pragma: no cover - assertion helper
        raise AssertionError(f"expected ValueError containing {contains!r}")


def _identity_rope(q, k, cos, sin, *args, **kwargs):
    return q, k


def test_base_capture_declares_q_window() -> None:
    signature = inspect.signature(ModelBackend.capture)
    assert "q_window" in signature.parameters
    assert signature.parameters["q_window"].default is None


def test_zero_q_window_owns_zero_storage() -> None:
    original_rope = llama_backend.modeling_llama.apply_rotary_pos_emb
    llama_backend.modeling_llama.apply_rotary_pos_emb = _identity_rope
    try:
        q = torch.randn(1, 4, 11, 8)
        k = torch.randn(1, 2, 11, 8)
        source_bytes = q.untyped_storage().nbytes()
        assert source_bytes > 0

        for to_cpu in (False, True):
            model = _FakeCaptureModel(n_layers=2, width=16)
            capture = llama_backend._QKVCapture(
                n_layers=2,
                n_kv_heads=2,
                head_dim=8,
                to_cpu=to_cpu,
                q_window=0,
            )
            with capture.attach(model):
                for layer in model.model.layers:
                    layer.self_attn(q, k)

            assert set(capture.layers) == {0, 1}
            for tensors in capture.layers.values():
                retained = tensors["q_post"]
                assert retained.shape == (1, 4, 0, 8)
                assert retained.untyped_storage().nbytes() == 0
                assert retained._base is None
    finally:
        llama_backend.modeling_llama.apply_rotary_pos_emb = original_rope


def test_cpu_q_slice_does_not_pin_full_storage() -> None:
    q = torch.randn(1, 4, 11, 8)
    capture = llama_backend._QKVCapture(
        n_layers=1,
        n_kv_heads=2,
        head_dim=8,
        to_cpu=True,
        q_window=2,
    )
    retained = capture._retain_q(q)
    assert retained.shape == (1, 4, 2, 8)
    assert retained.is_contiguous()
    assert retained.untyped_storage().nbytes() == retained.numel() * retained.element_size()
    assert retained.untyped_storage().data_ptr() != q.untyped_storage().data_ptr()


def test_capture_rejects_missing_and_duplicate_layers() -> None:
    original_rope = llama_backend.modeling_llama.apply_rotary_pos_emb
    llama_backend.modeling_llama.apply_rotary_pos_emb = _identity_rope
    q = torch.randn(1, 4, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    try:
        model = _FakeCaptureModel(n_layers=2, width=16)
        capture = llama_backend._QKVCapture(2, 2, 8, to_cpu=False)

        def missing_layer():
            with capture.attach(model):
                model.model.layers[0].self_attn(q, k)

        _expect_runtime_error("attention forwards=0", missing_layer)

        capture = llama_backend._QKVCapture(2, 2, 8, to_cpu=False)

        def duplicate_layer():
            with capture.attach(model):
                model.model.layers[0].self_attn(q, k)
                model.model.layers[0].self_attn(q, k)

        _expect_runtime_error("ran attention more than once", duplicate_layer)
    finally:
        llama_backend.modeling_llama.apply_rotary_pos_emb = original_rope


class _NeverCalledModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.config = types.SimpleNamespace(
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            hidden_size=8,
            head_dim=4,
        )

    def forward(self, *args, **kwargs):  # pragma: no cover - must stay unused
        raise AssertionError("model must not run when max_new <= 0")


def test_nonpositive_max_new_does_not_run_model() -> None:
    tokenizer = types.SimpleNamespace(eos_token_id=2)
    backend = llama_backend.LlamaBackend(_NeverCalledModel(), tokenizer)
    invalid_batch_is_intentional = torch.zeros(2, 3, dtype=torch.long)
    assert backend.sparse_generate(invalid_batch_is_intentional, 0, None) == ""
    assert backend.sparse_generate(invalid_batch_is_intentional, -3, None) == ""


def test_positive_generation_rejects_non_batch_one() -> None:
    tokenizer = types.SimpleNamespace(eos_token_id=2)
    backend = llama_backend.LlamaBackend(_NeverCalledModel(), tokenizer)
    invalid_batch = torch.zeros(2, 3, dtype=torch.long)
    _expect_value_error(
        "batch size 1",
        lambda: backend.sparse_generate(invalid_batch, 1, None),
    )


def test_backend_rejects_multiple_model_devices() -> None:
    class TwoDeviceModel(_NeverCalledModel):
        def __init__(self):
            super().__init__()
            self.register_buffer("meta_buffer", torch.empty(1, device="meta"))

    tokenizer = types.SimpleNamespace(eos_token_id=2)
    _expect_value_error(
        "single model device",
        lambda: llama_backend.LlamaBackend(TwoDeviceModel(), tokenizer),
    )


def test_load_forwards_revision_and_reports_provenance() -> None:
    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 2
        name_or_path = "tokenizer-source"

    class TokenizerLoader:
        calls = []

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            cls.calls.append((path, kwargs))
            return FakeTokenizer()

    class ModelLoader:
        calls = []

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            cls.calls.append((path, kwargs))
            model = _NeverCalledModel()
            model.config._name_or_path = "config-source"
            model.config._commit_hash = "0123456789abcdef"
            return model

    original_tokenizer = llama_backend.AutoTokenizer
    original_model = llama_backend.AutoModelForCausalLM
    llama_backend.AutoTokenizer = TokenizerLoader
    llama_backend.AutoModelForCausalLM = ModelLoader
    try:
        backend = llama_backend.LlamaBackend.load(
            "/tmp/local-model",
            dtype=torch.float32,
            device_map="cpu",
            revision="requested-tag",
        )
    finally:
        llama_backend.AutoTokenizer = original_tokenizer
        llama_backend.AutoModelForCausalLM = original_model

    assert TokenizerLoader.calls == [
        ("/tmp/local-model", {"revision": "requested-tag"})
    ]
    model_path, model_kwargs = ModelLoader.calls[0]
    assert model_path == "/tmp/local-model"
    assert model_kwargs["revision"] == "requested-tag"
    assert model_kwargs["torch_dtype"] is torch.float32
    assert model_kwargs["device_map"] == "cpu"

    provenance = backend.provenance()
    assert provenance == {
        "config_name_or_path": "config-source",
        "config_commit_hash": "0123456789abcdef",
        "tokenizer_name_or_path": "tokenizer-source",
        "requested_revision": "requested-tag",
        "resolved_revision": "0123456789abcdef",
        "model_dtype": "torch.float32",
        "model_device": "cpu",
        "prefill_mlp_chunk_size": None,
        "dims": {
            "n_layers": 1,
            "n_heads": 2,
            "n_kv_heads": 1,
            "head_dim": 4,
        },
    }

    # A requested tag or local path is not itself proof of resolution.
    backend.model.config._commit_hash = None
    unresolved = backend.provenance()
    assert unresolved["requested_revision"] == "requested-tag"
    assert unresolved["resolved_revision"] is None
    assert unresolved["config_commit_hash"] is None


def main() -> None:
    test_base_capture_declares_q_window()
    test_zero_q_window_owns_zero_storage()
    test_cpu_q_slice_does_not_pin_full_storage()
    test_capture_rejects_missing_and_duplicate_layers()
    test_nonpositive_max_new_does_not_run_model()
    test_positive_generation_rejects_non_batch_one()
    test_backend_rejects_multiple_model_devices()
    test_load_forwards_revision_and_reports_provenance()
    print("capture storage tests passed")


if __name__ == "__main__":
    main()
