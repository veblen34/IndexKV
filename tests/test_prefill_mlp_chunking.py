import unittest

import torch
from torch import nn

from indexkv.backends.llama import _ChunkedPrefillMLP


class _RecordingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden_states):
        self.calls.append(int(hidden_states.shape[-2]))
        return hidden_states * 2 + 1


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _RecordingMLP()


class _Backbone(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = _Backbone(layers)


class PrefillMLPChunkingTests(unittest.TestCase):
    def test_chunks_token_dimension_and_restores_original_forward(self):
        model = _Model([_Layer(), _Layer()])
        hidden_states = torch.arange(28, dtype=torch.float32).reshape(1, 7, 4)
        originals = [layer.mlp.forward for layer in model.model.layers]

        with _ChunkedPrefillMLP(3).attach(model):
            outputs = [layer.mlp(hidden_states) for layer in model.model.layers]

        for layer, output, original in zip(
            model.model.layers, outputs, originals
        ):
            self.assertTrue(torch.equal(output, hidden_states * 2 + 1))
            self.assertEqual(layer.mlp.calls, [3, 3, 1])
            self.assertEqual(layer.mlp.forward, original)

    def test_disabled_chunking_is_a_noop(self):
        model = _Model([_Layer()])
        original = model.model.layers[0].mlp.forward

        with _ChunkedPrefillMLP(None).attach(model):
            self.assertEqual(model.model.layers[0].mlp.forward, original)

    def test_rejects_nonpositive_chunk_size(self):
        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _ChunkedPrefillMLP(value)


if __name__ == "__main__":
    unittest.main()
