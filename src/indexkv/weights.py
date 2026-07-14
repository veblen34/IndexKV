"""Pretrained index-weight loading for HATA and HashAttention.

Both use trained tensors that are part of the index, so they are inside the
comparison scope. Loaders return ``{layer: weights}`` mappings that the method
code consumes directly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import torch


WEIGHTS_ROOT = Path(__file__).resolve().parents[2] / "weights"


class WeightsError(RuntimeError):
    """Raised when a method's pretrained weights cannot be loaded."""


def _norm(s: str) -> str:
    """Normalize model names for conservative artifact-directory matching."""
    s = s.lower().replace("meta-", "")
    for tok in ("instruct", "chat", "-hf"):
        s = s.replace(tok, "")
    return re.sub(r"[^a-z0-9]", "", s)


def _match_dir(candidates, model_name: str, what: str) -> Path:
    key = _norm(model_name)
    hits = [
        candidate
        for candidate in candidates
        if _norm(candidate.name)
        and (_norm(candidate.name) in key or key in _norm(candidate.name))
    ]
    if len(hits) == 1:
        return hits[0]
    names = [candidate.name for candidate in candidates]
    raise WeightsError(
        f"cannot resolve {what} for model '{model_name}': "
        f"{'ambiguous' if hits else 'no'} match among {names}. "
        "Pass an explicit path via --set <method>.weights_path=..."
    )


def _lfs_check(path: Path):
    if not path.is_file():
        raise WeightsError(f"weight artifact is not a file: {path}")
    if path.stat().st_size < 1024:
        head = path.read_bytes()[:40]
        if head.startswith(b"version https://git-lfs"):
            raise WeightsError(
                f"{path} is a git-lfs pointer, not the actual weights — run "
                "`git lfs pull` in the source repo and re-copy the file."
            )


def _resolve_hata_dir(
    model_name: str,
    *,
    rbits: int,
    root: Optional[Path],
    explicit_path: Optional[str],
) -> Path:
    if explicit_path:
        directory = Path(explicit_path)
    else:
        base = (root or WEIGHTS_ROOT) / "hata"
        candidates = (
            [
                path
                for path in base.iterdir()
                if path.is_dir() and path.name.endswith(f"-{rbits}")
            ]
            if base.is_dir()
            else []
        )
        if not candidates:
            raise WeightsError(
                f"no hata weight dirs under {base} for rbits={rbits}"
            )
        directory = _match_dir(
            candidates, model_name, f"hata weights (rbits={rbits})"
        )
    if not directory.is_dir():
        raise WeightsError(f"hata weights path is not a directory: {directory}")
    return directory


def _resolve_hashattention_file(
    model_name: str,
    *,
    root: Optional[Path],
    explicit_path: Optional[str],
) -> Path:
    if explicit_path:
        artifact = Path(explicit_path)
    else:
        base = (root or WEIGHTS_ROOT) / "hashattention"
        candidates = list(base.glob("*.pt")) if base.is_dir() else []
        if not candidates:
            raise WeightsError(f"no hashattention .pt patches under {base}")
        artifact = _match_dir(candidates, model_name, "hashattention patch")
    _lfs_check(artifact)
    return artifact


def load_hata_weights(
    model_name: str,
    *,
    rbits: int = 128,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    root: Optional[Path] = None,
    explicit_path: Optional[str] = None,
) -> Dict[int, torch.Tensor]:
    """Load ``{layer: (H_kv, head_dim, rbits)}`` HATA projections."""
    directory = _resolve_hata_dir(
        model_name,
        rbits=rbits,
        root=root,
        explicit_path=explicit_path,
    )
    files = sorted(directory.glob("hash_weight_layer_*.pt"))
    if not files:
        raise WeightsError(f"no hash_weight_layer_*.pt files in {directory}")
    for artifact in files:
        _lfs_check(artifact)

    out: Dict[int, torch.Tensor] = {}
    for artifact in files:
        try:
            layer_idx = int(artifact.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise WeightsError(
                f"cannot parse HATA layer id from {artifact.name}"
            ) from exc
        if layer_idx in out:
            raise WeightsError(f"duplicate HATA artifact for layer {layer_idx}")
        weight = torch.load(artifact, map_location="cpu", weights_only=True)
        if not isinstance(weight, torch.Tensor):
            raise WeightsError(f"{artifact} did not contain one tensor")
        out[layer_idx] = weight.to(device=device, dtype=dtype)
    return out


def load_hashattention_weights(
    model_name: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    root: Optional[Path] = None,
    explicit_path: Optional[str] = None,
) -> Dict[int, Dict[str, list]]:
    """Parse a USA ModuleList state dict into per-layer stacked MLPs.

    State-dict keys are
    ``{L}.learning_to_hash_transformation_{k|q}.{head}.{i}.{weight|bias}``
    where Sequential indices ``i`` identify the three Linear stages.
    """
    artifact = _resolve_hashattention_file(
        model_name, root=root, explicit_path=explicit_path
    )
    state_dict = torch.load(artifact, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise WeightsError(f"{artifact} did not contain a state dict")

    pattern = re.compile(
        r"^(\d+)\.learning_to_hash_transformation_([kq])\."
        r"(\d+)\.(\d+)\.(weight|bias)$"
    )
    tree: dict = {}
    for key, tensor in state_dict.items():
        match = pattern.match(key)
        if not match:
            continue
        layer_idx = int(match[1])
        kq = match[2]
        head = int(match[3])
        sequential_idx = int(match[4])
        weight_or_bias = match[5]
        tree.setdefault(layer_idx, {}).setdefault(kq, {}).setdefault(
            sequential_idx, {}
        ).setdefault(weight_or_bias, {})[head] = tensor
    if not tree:
        raise WeightsError(f"{artifact} contains no USA keys")

    out: Dict[int, Dict[str, list]] = {}
    for layer_idx, kqs in tree.items():
        entry = {}
        for kq, sequences in kqs.items():
            linears = []
            for sequential_idx in sorted(sequences):
                fields = sequences[sequential_idx]
                if set(fields) != {"weight", "bias"}:
                    raise WeightsError(
                        f"{artifact} layer {layer_idx} {kq} stage "
                        f"{sequential_idx} lacks weight or bias"
                    )
                weight_heads = set(fields["weight"])
                bias_heads = set(fields["bias"])
                if weight_heads != bias_heads:
                    raise WeightsError(
                        f"{artifact} layer {layer_idx} {kq} stage "
                        f"{sequential_idx} has mismatched weight/bias heads"
                    )
                heads = sorted(weight_heads)
                if heads != list(range(len(heads))):
                    raise WeightsError(
                        f"{artifact} layer {layer_idx} {kq} stage "
                        f"{sequential_idx} has non-contiguous heads {heads}"
                    )
                weight = torch.stack([fields["weight"][head] for head in heads])
                bias = torch.stack([fields["bias"][head] for head in heads])
                linears.append(
                    (
                        weight.to(device=device, dtype=dtype),
                        bias.to(device=device, dtype=dtype),
                    )
                )
            entry[kq] = linears
        out[layer_idx] = entry
    return out


def load_method_weights(
    weights_key: str,
    model_name: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    extra: Optional[dict] = None,
) -> Dict[int, object]:
    """Load one method's artifacts; returned mapping may expose ``artifacts``."""
    extra = extra or {}
    path = extra.get("weights_path")
    if weights_key == "hata":
        return load_hata_weights(
            model_name,
            rbits=int(extra.get("rbits", 128)),
            device=device,
            dtype=dtype,
            explicit_path=path,
        )
    if weights_key == "hashattention":
        return load_hashattention_weights(
            model_name,
            device=device,
            dtype=dtype,
            explicit_path=path,
        )
    raise WeightsError(f"unknown weights_key '{weights_key}'")
