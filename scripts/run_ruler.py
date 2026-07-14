"""RULER sweep: methods x budgets x tasks, registry-driven.

The complete RULER prompt is fed raw (no chat template), decoding is greedy, and
the official task-category string-match scorer is used. Every method shares the
same capture, forced sink/recent windows, dense prefix, and token budget, so the
only thing that varies is index-selection quality.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# make src/ importable without an install
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from indexkv.backends.llama import LlamaBackend
except ModuleNotFoundError as exc:  # allow --list_methods without model deps
    LlamaBackend = None
    _llama_import_error = exc
else:
    _llama_import_error = None

from indexkv.runner import SweepRunner, parse_set_overrides, print_method_listing
from indexkv.registry import list_methods
from indexkv.eval.ruler import (
    RULER_TASKS,
    build_ruler_prompt,
    load_ruler,
    max_new_tokens,
    ruler_task_score,
)
from indexkv.manifest import (
    atomic_write_json,
    atomic_write_jsonl,
    collect_environment,
    set_seed,
    utc_timestamp,
)


def parse_args():
    p = argparse.ArgumentParser(description="RULER index-only sweep")
    p.add_argument("--model", default=None, help="model path (required unless --list_methods)")
    p.add_argument("--model_name", default=None,
                   help="name used to resolve trained index weights; defaults to basename of --model")
    p.add_argument("--tasks", nargs="+", default=RULER_TASKS)
    p.add_argument("--methods", nargs="+", default=["full"])
    p.add_argument("--budgets", nargs="+", type=int, default=[256, 512, 1024, 2048])
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--data_root", default=None,
                   help="root containing <task>/validation.jsonl (required for runs)")
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--sink", type=int, default=4)
    p.add_argument("--recent", type=int, default=32)
    p.add_argument("--dense_prefix_layers", type=int, default=2,
                   help="global dense prefix applied identically to every method")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--out_dir", default="results/ruler")
    p.add_argument("--set", nargs="*", default=[],
                   help="method-scoped knobs, e.g. --set hata.rbits=256 chunkkv.chunk_length=8")
    p.add_argument("--list_methods", action="store_true")
    return p.parse_args()


def _duplicates(values) -> list:
    seen, dup = set(), set()
    for v in values:
        if v in seen:
            dup.add(v)
        seen.add(v)
    return sorted(dup)


def _validate_args(args) -> None:
    for flag, values in (("--tasks", args.tasks), ("--methods", args.methods),
                         ("--budgets", args.budgets)):
        if _duplicates(values):
            raise SystemExit(f"{flag} contains duplicates: {_duplicates(values)}")
    unknown = sorted(set(args.tasks) - set(RULER_TASKS))
    if unknown:
        raise SystemExit(f"--tasks contains unsupported tasks: {unknown}")
    if any(b < 0 for b in args.budgets):
        raise SystemExit(f"--budgets must be >= 0, got {args.budgets}")
    if args.num_samples is not None and args.num_samples <= 0:
        raise SystemExit("--num_samples must be > 0 when provided")
    if not isinstance(args.data_root, str) or not args.data_root.strip():
        raise SystemExit("--data_root is required for a RULER run")
    if args.block_size <= 0:
        raise SystemExit("--block_size must be > 0")
    if args.sink < 0 or args.recent < 0 or args.dense_prefix_layers < 0:
        raise SystemExit("--sink, --recent, --dense_prefix_layers must be >= 0")


def _validate_empty_out_dir(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise SystemExit(f"--out_dir exists and is not a directory: {path}")
    if next(path.iterdir(), None) is not None:
        raise SystemExit(f"--out_dir must be absent or empty; refusing to overwrite {path}")


def _load_backend(args, dtype):
    if LlamaBackend is None:
        raise SystemExit(
            f"Llama backend dependencies are unavailable ({_llama_import_error})"
        )
    return LlamaBackend.load(args.model, dtype=dtype)


def selection_summary(stats) -> dict:
    agg = (stats or {}).get("selection_aggregate", {})
    mins, maxs, means = [], [], []
    for v in agg.values():
        sc = int(v.get("sample_count", 0))
        if sc:
            mins.append(int(v["min"]))
            maxs.append(int(v["max"]))
            means.append(int(v["sum"]) / sc)
    return {
        "layers_scored": len(means),
        "selected_min": min(mins) if mins else None,
        "selected_max": max(maxs) if maxs else None,
        "selected_mean": round(sum(means) / len(means), 2) if means else None,
        "selection_s_total": float((stats or {}).get("selection_s_total", 0.0)),
    }


def _synchronize_for(ids) -> None:
    device = getattr(ids, "device", None)
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_capture(runner, ids):
    _synchronize_for(ids)
    start = time.perf_counter()
    capture = runner.capture(ids)
    _synchronize_for(ids)
    return capture, time.perf_counter() - start


def _timing(capture_s, build_s, gen_timing) -> dict:
    return {
        "capture_s": float(capture_s),
        "index_build_s": float(build_s),
        "index_query_s": float(gen_timing.get("selection_s_total", 0.0)),
        "generation_s": float(gen_timing.get("model_decode_s_exclusive", 0.0)),
        "total_s": float(gen_timing.get("total_s", 0.0)),
    }


def main():
    args = parse_args()
    if args.list_methods:
        print_method_listing()
        return
    if args.model is None:
        raise SystemExit("--model is required (or use --list_methods)")

    _validate_args(args)
    out_dir = Path(args.out_dir)
    _validate_empty_out_dir(out_dir)
    seed_info = set_seed(args.seed, args.deterministic)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    model_name = args.model_name or Path(args.model).name
    overrides = parse_set_overrides(args.set)
    backend = _load_backend(args, dtype)

    runner = SweepRunner(
        backend, args.methods, model_name=model_name,
        block_size=args.block_size, sink=args.sink, recent=args.recent,
        dense_prefix_layers=args.dense_prefix_layers, dtype=dtype, overrides=overrides,
    )
    for m, reason in runner.skipped.items():
        print(f"[skip] {m}: {reason}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    all_record_files: dict = {}
    total_processed = total_skipped_oom = 0
    run_t0 = time.time()

    for task in args.tasks:
        max_new = max_new_tokens(task)
        samples = load_ruler(task, data_root=args.data_root)
        n = len(samples) if args.num_samples is None else min(args.num_samples, len(samples))

        records: dict[str, list[dict]] = {}
        build_s_by_method: dict[str, list[float]] = {}
        processed = skipped_oom = 0
        t0 = time.time()

        for si in range(n):
            sample = samples[si]
            # kvpress-consistent RULER prompt: wrap [context+question] in the model
            # chat template, answer prefix after the assistant header (not raw text).
            full = build_ruler_prompt(task, sample["input"], backend.tok)
            ids = backend.tok(
                full, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(next(backend.model.parameters()).device)
            prompt_tokens = int(ids.shape[1])
            staged_records: dict[str, list[dict]] = {}
            staged_build: dict[str, float] = {}
            cap = None
            try:
                cap, capture_s = _timed_capture(runner, ids)
                for method_name in runner.runnable:
                    indices, build_elapsed = runner.build(method_name, cap)
                    staged_build[method_name] = build_elapsed
                    budgets = [None] if method_name == "full" else args.budgets
                    for budget in budgets:
                        key = method_name if budget is None else f"{method_name}/bud={budget}"
                        pred = runner.generate(method_name, indices, cap, budget, ids, max_new)
                        timing = _timing(capture_s, build_elapsed, dict(runner.last_generation_timing))
                        staged_records.setdefault(key, []).append({
                            "task": task,
                            "result_key": key,
                            "method": method_name,
                            "budget": budget,
                            "sample_position": si,
                            "sample_index": sample.get("index", si),
                            "length": sample.get("length"),
                            "prompt_tokens": prompt_tokens,
                            "pred": pred,
                            "refs": sample["outputs"],
                            "selection": selection_summary(runner.last_selection_stats),
                            "timing": timing,
                        })
            except torch.cuda.OutOfMemoryError:
                skipped_oom += 1
                print(f"[oom] skipped {task}/{sample.get('index', si)} (prompt tokens={prompt_tokens})")
                torch.cuda.empty_cache()
                continue
            finally:
                cap = None

            for method_name, elapsed in staged_build.items():
                build_s_by_method.setdefault(method_name, []).append(elapsed)
            for key, rows in staged_records.items():
                records.setdefault(key, []).extend(rows)
            processed += 1
            torch.cuda.empty_cache()

        agg: dict = {}
        for m, reason in runner.skipped.items():
            agg[m] = {"status": reason.split(":", 1)[0], "detail": reason}
        record_files: dict = {}
        for key, rows in records.items():
            method = key.split("/", 1)[0]
            preds = [r["pred"] for r in rows]
            refs = [r["refs"] for r in rows]
            build_s = build_s_by_method.get(method, [0.0])
            agg[key] = {
                "status": "ok",
                "score": ruler_task_score(task, preds, refs),
                "n": len(rows),
                "gen_s_mean": round(sum(r["timing"]["generation_s"] for r in rows) / len(rows), 4),
                "index_build_s_mean": round(sum(build_s) / len(build_s), 4),
            }
            safe = key.replace("/", "__")
            filename = f"pred_{task}__{safe}.jsonl"
            atomic_write_jsonl(out_dir / filename, rows)
            record_files[key] = filename

        summary[task] = agg
        all_record_files[task] = record_files
        total_processed += processed
        total_skipped_oom += skipped_oom

        atomic_write_json(out_dir / f"{task}.json", {
            "benchmark": "ruler",
            "task": task,
            "timestamp_utc": utc_timestamp(),
            "elapsed_s": time.time() - t0,
            "processed": processed,
            "skipped_oom": skipped_oom,
            "dataset_size": len(samples),
            "requested_samples": n,
            "record_files": record_files,
            "agg": agg,
        })
        print(f"[{task}] " + json.dumps(agg))

    manifest = {
        "benchmark": "ruler",
        "timestamp_utc": utc_timestamp(),
        "environment": collect_environment(),
        "seed": seed_info,
        "args": vars(args),
        "resolved_model_name": model_name,
        "data_root": args.data_root,
        "elapsed_s": time.time() - run_t0,
        "processed": total_processed,
        "skipped_oom": total_skipped_oom,
        "methods": {
            "requested": list(args.methods),
            "runnable": list(runner.runnable),
            "skipped": dict(runner.skipped),
            "budgets": list(args.budgets),
            "overrides": overrides,
            "metadata": {n: list_methods()[n] for n in args.methods if n in list_methods()},
        },
        "record_files": all_record_files,
        "tasks": summary,
    }
    atomic_write_json(out_dir / "_summary.json", manifest)
    print(f"\nwrote {out_dir}/_summary.json")


if __name__ == "__main__":
    main()
