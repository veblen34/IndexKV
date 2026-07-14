"""LongBench v2 sweep: methods x budgets on the official 0-shot protocol.

For each sample the prompt's post-RoPE K/V/Q are captured once; then every
method builds its index and generates under each token budget from that shared
capture, so the only thing that varies across methods is index-selection
quality. Prompt rendering, middle-truncation, answer extraction and scoring all
follow the official THUDM/LongBench-v2 ``pred.py`` / ``result.py``.

One sample is a transaction: its records are committed only after every
method/budget for that sample finishes, so an OOM on a long prompt skips the
whole sample cleanly and the sweep continues.
"""

from __future__ import annotations

import argparse
import inspect
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
from indexkv.manifest import (
    atomic_write_json,
    atomic_write_jsonl,
    collect_environment,
    set_seed,
    utc_timestamp,
)
from indexkv.eval.longbench2 import (
    build_prompt,
    compare_to_leaderboard,
    extract_answer,
    iter_official_items,
    score_predictions,
    truncate_prompt,
)


def parse_args():
    p = argparse.ArgumentParser(description="LongBench v2 index-only sweep")
    p.add_argument("--model", default=None, help="model path (required unless --list_methods)")
    p.add_argument("--model_name", default="Llama-3.1-8B-Instruct",
                   help="key into config/model2maxlen.json; also resolves trained index weights")
    p.add_argument("--methods", nargs="+", default=["full"])
    p.add_argument("--budgets", nargs="+", type=int, default=[512, 1024, 2048])
    p.add_argument("--num_samples", type=int, default=None,
                   help="stop after N *processed* samples (skipped ones do not count)")
    p.add_argument("--max_prompt_tokens", type=int, default=None,
                   help="skip samples whose truncated prompt exceeds this many tokens "
                        "(hardware filter to avoid OOM on very long prompts)")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--sink", type=int, default=4)
    p.add_argument("--recent", type=int, default=32)
    p.add_argument("--dense_prefix_layers", type=int, default=2,
                   help="global dense prefix applied identically to every method")
    p.add_argument("--prefill_mlp_chunk_size", type=int, default=None,
                   help="optional token chunk size for prefill MLP; bounds memory "
                        "without changing attention/KV/selection")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--out_dir", default="results/longbench2")
    p.add_argument("--resume", action="store_true",
                   help="continue a prior run in --out_dir, skipping already-done samples")
    p.add_argument("--checkpoint_every", type=int, default=10,
                   help="flush prediction files to disk every N processed samples")
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
    if _duplicates(args.methods):
        raise SystemExit(f"--methods contains duplicates: {_duplicates(args.methods)}")
    if _duplicates(args.budgets):
        raise SystemExit(f"--budgets contains duplicates: {_duplicates(args.budgets)}")
    if any(b < 0 for b in args.budgets):
        raise SystemExit(f"--budgets must be >= 0, got {args.budgets}")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max_new_tokens must be > 0")
    if args.num_samples is not None and args.num_samples <= 0:
        raise SystemExit("--num_samples must be > 0 when provided")
    if args.max_prompt_tokens is not None and args.max_prompt_tokens <= 0:
        raise SystemExit("--max_prompt_tokens must be > 0 when provided")
    if args.block_size <= 0:
        raise SystemExit("--block_size must be > 0")
    if args.prefill_mlp_chunk_size is not None and args.prefill_mlp_chunk_size <= 0:
        raise SystemExit("--prefill_mlp_chunk_size must be > 0 when provided")
    if args.sink < 0 or args.recent < 0 or args.dense_prefix_layers < 0:
        raise SystemExit("--sink, --recent, --dense_prefix_layers must be >= 0")
    if args.checkpoint_every <= 0:
        raise SystemExit("--checkpoint_every must be > 0")


def _validate_empty_out_dir(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise SystemExit(f"--out_dir exists and is not a directory: {path}")
    if next(path.iterdir(), None) is not None:
        raise SystemExit(f"--out_dir must be absent or empty; refusing to overwrite {path}")


def _append_skip(out_dir: Path, event: dict) -> None:
    """Record a skipped sample immediately so resume never re-attempts it."""
    with (out_dir / "_skipped.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
        f.flush()


def _flush_records(out_dir: Path, records: dict) -> dict:
    """Atomically (re)write every cell's prediction file. Durable checkpoint."""
    record_files = {}
    for key, recs in records.items():
        safe = key.replace("/", "__")
        path = out_dir / f"pred_{safe}.jsonl"
        atomic_write_jsonl(path, recs)
        record_files[key] = path.name
    return record_files


def _load_resume_state(out_dir: Path):
    """Rebuild in-memory state from a prior run's checkpoint files.

    A processed sample is one present in the flushed pred files (all cells are
    flushed together, so they agree). Skipped samples come from _skipped.jsonl.
    """
    records: dict[str, list[dict]] = {}
    processed_ids: set[str] = set()
    for path in sorted(out_dir.glob("pred_*.jsonl")):
        key = path.stem[len("pred_"):].replace("__", "/")
        recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        records[key] = recs
        processed_ids.update(r["_id"] for r in recs)
    ref = records.get("full") or (next(iter(records.values())) if records else [])
    proc_events = {
        r["_id"]: {"_id": r["_id"], "prompt_tokens": r.get("prompt_tokens"),
                   "difficulty": r.get("difficulty"), "length": r.get("length"),
                   "domain": r.get("domain"), "status": "processed"}
        for r in ref
    }
    skip_by_id: dict = {}
    skip_path = out_dir / "_skipped.jsonl"
    if skip_path.exists():
        for line in skip_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                if e["_id"] not in processed_ids:
                    skip_by_id[e["_id"]] = e
    sample_events = list(proc_events.values()) + list(skip_by_id.values())
    done_ids = set(processed_ids) | set(skip_by_id)
    processed = len(proc_events)
    skipped_len = sum(1 for e in skip_by_id.values() if e["status"] == "skipped_len")
    skipped_oom = sum(1 for e in skip_by_id.values() if e["status"] == "skipped_oom")
    return records, sample_events, done_ids, processed, skipped_len, skipped_oom


def _load_backend(args, dtype):
    if LlamaBackend is None:
        raise SystemExit(
            f"Llama backend dependencies are unavailable ({_llama_import_error})"
        )
    kwargs = {"dtype": dtype}
    if args.prefill_mlp_chunk_size is not None:
        params = inspect.signature(LlamaBackend.load).parameters
        if "prefill_mlp_chunk_size" not in params:
            raise SystemExit("backend.load has no prefill_mlp_chunk_size parameter")
        kwargs["prefill_mlp_chunk_size"] = args.prefill_mlp_chunk_size
    return LlamaBackend.load(args.model, **kwargs)


def selection_summary(stats) -> dict:
    """Slim, human-readable check that selection stayed within budget."""
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


def build_ids(backend, prompt_str, model_name):
    """Official order: middle-truncate the rendered prompt, then chat template."""
    truncated = truncate_prompt(prompt_str, model_name, backend.tok)
    return backend.tokenize(truncated, chat_template=True)


def main():
    args = parse_args()
    if args.list_methods:
        print_method_listing()
        return
    if args.model is None:
        raise SystemExit("--model is required (or use --list_methods)")

    _validate_args(args)
    out_dir = Path(args.out_dir)
    if not args.resume:
        _validate_empty_out_dir(out_dir)
    seed_info = set_seed(args.seed, args.deterministic)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    overrides = parse_set_overrides(args.set)
    backend = _load_backend(args, dtype)

    runner = SweepRunner(
        backend, args.methods, model_name=args.model_name,
        block_size=args.block_size, sink=args.sink, recent=args.recent,
        dense_prefix_layers=args.dense_prefix_layers, dtype=dtype, overrides=overrides,
    )
    for m, reason in runner.skipped.items():
        print(f"[skip] {m}: {reason}")

    items = iter_official_items(split="train")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        (records, sample_events, done_ids, processed,
         skipped_len, skipped_oom) = _load_resume_state(out_dir)
        print(f"[resume] {len(done_ids)} samples already done "
              f"(processed={processed} skipped_len={skipped_len} skipped_oom={skipped_oom})")
    else:
        records, sample_events, done_ids = {}, [], set()
        processed = skipped_len = skipped_oom = 0
    t0 = time.time()

    def make_record(item, response, *, method_name, budget, prompt_tokens,
                    result_key, selection, timing):
        response = response.strip()
        pred = extract_answer(response)
        return {
            "_id": item["_id"],
            "difficulty": item["difficulty"],
            "length": item["length"],
            "domain": item["domain"],
            "answer": item["answer"],
            "pred": pred,
            "judge": pred == item["answer"],
            "response": response,
            "method": method_name,
            "budget": budget,
            "prompt_tokens": int(prompt_tokens),
            "result_key": result_key,
            "selection": selection,
            "timing": timing,
        }

    def _run_item(item, ids):
        staged: dict[str, list[dict]] = {}
        base_prompt_tokens = int(ids.shape[1])
        cap = None
        try:
            cap, capture_s = _timed_capture(runner, ids)
            for method_name in runner.runnable:
                indices, build_s = runner.build(method_name, cap)
                budgets = [None] if method_name == "full" else args.budgets
                for budget in budgets:
                    key = method_name if budget is None else f"{method_name}/bud={budget}"
                    response = runner.generate(
                        method_name, indices, cap, budget, ids, args.max_new_tokens)
                    timing = _timing(capture_s, build_s, dict(runner.last_generation_timing))
                    selection = selection_summary(runner.last_selection_stats)
                    staged.setdefault(key, []).append(make_record(
                        item, response, method_name=method_name, budget=budget,
                        prompt_tokens=base_prompt_tokens, result_key=key,
                        selection=selection, timing=timing))
        finally:
            cap = None
        return staged

    for item in items:
        if args.num_samples is not None and processed >= args.num_samples:
            break
        if item["_id"] in done_ids:
            continue
        ids = build_ids(backend, build_prompt(item), args.model_name)
        prompt_tokens = int(ids.shape[1])
        event = {"_id": item["_id"], "prompt_tokens": prompt_tokens,
                 "difficulty": item["difficulty"], "length": item["length"],
                 "domain": item["domain"]}

        if args.max_prompt_tokens is not None and prompt_tokens > args.max_prompt_tokens:
            skipped_len += 1
            ev = {**event, "status": "skipped_len"}
            sample_events.append(ev)
            done_ids.add(item["_id"])
            _append_skip(out_dir, ev)
            continue
        try:
            staged = _run_item(item, ids)
        except torch.cuda.OutOfMemoryError:
            skipped_oom += 1
            ev = {**event, "status": "skipped_oom"}
            sample_events.append(ev)
            done_ids.add(item["_id"])
            _append_skip(out_dir, ev)
            print(f"[oom] skipped {item['_id']} (prompt tokens={prompt_tokens})", flush=True)
            torch.cuda.empty_cache()
            continue

        for key, staged_records in staged.items():
            records.setdefault(key, []).extend(staged_records)
        processed += 1
        sample_events.append({**event, "status": "processed"})
        done_ids.add(item["_id"])
        if processed % args.checkpoint_every == 0:
            _flush_records(out_dir, records)
            print(f"[checkpoint] processed={processed} skipped_oom={skipped_oom} "
                  f"flushed -> {out_dir}", flush=True)
        torch.cuda.empty_cache()

    print(f"processed={processed} skipped_len={skipped_len} skipped_oom={skipped_oom}")

    full_run = (args.num_samples is None and args.max_prompt_tokens is None
                and skipped_len == 0 and skipped_oom == 0 and processed == len(items))

    summary: dict = {}
    record_files: dict = {}
    for m, reason in runner.skipped.items():
        summary[m] = {"status": reason.split(":", 1)[0], "detail": reason}
    for key, recs in records.items():
        score = score_predictions(recs)
        entry = {"status": "ok", **score.as_dict()}
        if full_run and args.model_name == "Llama-3.1-8B-Instruct":
            try:
                entry["leaderboard_delta"] = compare_to_leaderboard(score, model=args.model_name)
            except KeyError:
                pass
        summary[key] = entry
        safe = key.replace("/", "__")
        path = out_dir / f"pred_{safe}.jsonl"
        atomic_write_jsonl(path, recs)
        record_files[key] = path.name

    manifest = {
        "benchmark": "longbench2",
        "timestamp_utc": utc_timestamp(),
        "environment": collect_environment(),
        "seed": seed_info,
        "args": vars(args),
        "elapsed_s": time.time() - t0,
        "processed": processed,
        "skipped_len": skipped_len,
        "skipped_oom": skipped_oom,
        "full_run": full_run,
        "methods": {
            "requested": list(args.methods),
            "runnable": list(runner.runnable),
            "skipped": dict(runner.skipped),
            "budgets": list(args.budgets),
            "overrides": overrides,
            "result_keys": sorted(records),
            "metadata": {n: list_methods()[n] for n in args.methods if n in list_methods()},
        },
        "samples": {
            "dataset_size": len(items),
            "seen": len(sample_events),
            "processed": processed,
            "items": sample_events,
        },
        "record_files": record_files,
        "results": summary,
    }
    atomic_write_json(out_dir / "_summary.json", manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_dir}/_summary.json")


if __name__ == "__main__":
    main()
