"""Small runtime utilities shared by the benchmark scripts.

Just enough to write results atomically, stamp a run with a timestamp and the
software/GPU it ran on, and set RNG seeds for reproducible decoding. No source
snapshots, git state, or provenance hashing — the framework's fairness is
enforced at runtime by the engine, not by an offline audit gate.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, TextIO


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp with an explicit ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict:
    """Record software and GPU versions (best effort, never raises)."""
    software = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "datasets": _package_version("datasets"),
    }
    accelerator: dict = {"cuda_available": False, "devices": []}
    try:
        import torch

        accelerator["cuda_available"] = bool(torch.cuda.is_available())
        accelerator["torch_cuda_version"] = torch.version.cuda
        if accelerator["cuda_available"]:
            devices = []
            for ordinal in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(ordinal)
                devices.append({
                    "ordinal": ordinal,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                })
            accelerator["devices"] = devices
    except Exception as exc:  # metadata collection must never abort a run
        accelerator["probe_error"] = f"{type(exc).__name__}: {exc}"
    return {"software": software, "accelerator": accelerator}


def set_seed(seed: int, deterministic: bool = False) -> dict:
    """Seed Python/NumPy/torch RNGs so greedy decoding is reproducible."""
    random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    try:
        import numpy as np

        np.random.seed(seed % 2**32)
    except ImportError:
        pass
    return {"seed": int(seed), "deterministic": bool(deterministic)}


def _atomic_dump(path: str | Path, writer: Callable[[TextIO], None]) -> None:
    """Write a text file beside its destination and atomically replace it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, value: object, *, ensure_ascii: bool = False) -> None:
    """Atomically write a human-readable JSON document."""
    def write(handle: TextIO) -> None:
        json.dump(value, handle, indent=2, ensure_ascii=ensure_ascii)
        handle.write("\n")

    _atomic_dump(path, write)


def atomic_write_jsonl(
    path: str | Path,
    records: Iterable[Mapping[str, object]],
    *,
    ensure_ascii: bool = False,
) -> None:
    """Atomically stream JSONL records without building one large string."""
    def write(handle: TextIO) -> None:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=ensure_ascii))
            handle.write("\n")

    _atomic_dump(path, write)
