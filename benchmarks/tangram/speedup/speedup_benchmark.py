#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark: Tangram KV-cache compression speedup vs. the uncompressed baseline.

Under a KV-pressured workload, compression shrinks each request's KV footprint
so more requests fit at once; fewer requests queue/preempt and end-to-end time
drops. This harness quantifies that speedup by running the same workload twice —
an uncompressed baseline (ratio 1.0, full KV cache) and a compressed run
(``--ratio`` < 1.0, prefill-with-eviction) — and printing a side-by-side
comparison of preemptions, end-to-end time, throughput, and the speedup.

``scheduler_reserve_full_isl`` is pinned ON for every run, so admission control
is held constant and the only varied axis is the compression ratio.

Compression defaults to the SnapKV scorer at the ``perlayer_cluster`` selection
level (matching the shipping library defaults); both are configurable via
``--compression-scorer`` / ``--compression-level``.

Examples
--------
    # Baseline (ratio 1.0) vs SnapKV at a 30% KV budget.
    CUDA_VISIBLE_DEVICES=0 python speedup_benchmark.py --ratio 0.3

    # Tighten the KV cache for more contention (larger speedup).
    CUDA_VISIBLE_DEVICES=0 python speedup_benchmark.py \\
        --ratio 0.3 --gpu-memory-utilization 0.12 --num 32

    # Override the scorer or selection level.
    CUDA_VISIBLE_DEVICES=0 python speedup_benchmark.py \\
        --ratio 0.3 --compression-scorer fastkvzip --compression-level perlayer_head

    # Run a single configuration only.
    CUDA_VISIBLE_DEVICES=0 python speedup_benchmark.py --run-ratio 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

# scbench data loaders live in the sibling tangram benchmark dir.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TANGRAM_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(os.path.dirname(_TANGRAM_DIR))
for _p in (_TANGRAM_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# Head-group cluster maps ship under
# tools/head_group_clustering/cluster_maps/<method>/<model-id>/pg<pg>_r<base>[_perlayer].npz
# A cluster-granularity selection level (crosslayer_cluster / perlayer_cluster)
# needs one; the head levels run fine on identity adjacency. We auto-resolve the
# map from (scorer, model, pg, level) and fall back to identity (None) when no
# map is on disk.
_CMAP_DIR = os.path.join(_REPO_ROOT, "tools", "head_group_clustering",
                         "cluster_maps")
_CMAP_BASE_RATIO = 0.3  # one base-ratio ranking map is valid across all ratios.
# Scorer -> method folder. Identity-mapped except expected_attention -> 'ea'.
_SCORER_CMAP_METHOD = {
    "fastkvzip": "fastkvzip", "snapkv": "snapkv", "keydiff": "keydiff",
    "expected_attention": "ea",
}
# Only the per-layer-scope levels use the _perlayer map flavour.
_LEVEL_CMAP_SUFFIX = {"perlayer_cluster": "_perlayer", "perlayer_head": "_perlayer"}


def resolve_cluster_map(args: argparse.Namespace) -> str | None:
    """Resolve the head-group cluster map path for this run.

    An explicit ``--head-group-cluster-map`` always wins. Otherwise auto-resolve
    from (scorer, model basename, page-group size, level); return None (identity
    adjacency) when the scorer has no map folder or the file is not on disk."""
    if args.head_group_cluster_map:
        return args.head_group_cluster_map
    method = _SCORER_CMAP_METHOD.get(args.compression_scorer)
    if method is None:
        return None
    model_id = os.path.basename(args.model.rstrip("/")).lower()
    suffix = _LEVEL_CMAP_SUFFIX.get(args.compression_level, "")
    path = os.path.join(_CMAP_DIR, method, model_id,
                        f"pg{args.page_group_size}_r{_CMAP_BASE_RATIO}{suffix}.npz")
    return path if os.path.exists(path) else None


# ---------------------------------------------------------------------------
# Worker: run ONE configuration (at a given ratio) and report its metrics.
# ---------------------------------------------------------------------------
def run_one(args: argparse.Namespace, run_ratio: float) -> dict:
    # Heavy imports kept inside the worker so the orchestrator stays light.
    from transformers import AutoTokenizer

    from benchmark_scbench import get_eval_task, get_query_text
    from scbench_local import load_dataset_all, template
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_basename = os.path.basename(args.model.rstrip("/"))
    task = get_eval_task(args.dataset)
    prefix, postfix = template(model_basename, args.dataset)

    dataset = load_dataset_all(args.dataset, tok, n_data=args.num)
    prompts, ctx_lens, skipped = [], [], 0
    for i in range(min(args.num, len(dataset))):
        questions = list(dataset[i]["question"])
        if not questions or not questions[0]:
            continue
        qtext = get_query_text(task, questions[0]).strip()
        ids = tok.encode(f"{prefix}{dataset[i]['context']}\n\n{qtext}{postfix}",
                         add_special_tokens=False)
        if len(ids) + args.max_tokens > args.max_model_len:
            skipped += 1
            continue
        prompts.append({"prompt_token_ids": ids})
        ctx_lens.append(len(ids))
    if not prompts:
        raise SystemExit("No usable prompts after length filtering.")
    if skipped:
        print(f"  (dropped {skipped} sample(s) longer than the context window)")

    llm_kwargs = dict(
        model=args.model,
        dtype="auto",
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        # Compression forces prefix caching off; keep it off in the baseline too
        # so baseline/compressed differ only in the KV budget.
        enable_prefix_caching=False,
        disable_log_stats=False,  # required so get_metrics() is populated
        # Pinned ON so admission control is held constant across both runs.
        scheduler_reserve_full_isl=True,
        watermark=args.watermark,
        # Head-grouped paging layout (pg=4 default).
        page_group_size=args.page_group_size,
        # Cluster-granularity levels need a cluster map; auto-resolved from
        # (scorer, model, pg, level). Only consulted for compressed runs below.
        head_group_cluster_map=(
            resolve_cluster_map(args) if run_ratio < 1.0 else None),
    )
    if run_ratio < 1.0:
        cmap = llm_kwargs["head_group_cluster_map"]
        print(f"  cluster map: {cmap if cmap else 'identity (none on disk)'}")
        if cmap is None and args.compression_level.endswith("_cluster"):
            print(f"  WARNING: level={args.compression_level} needs a cluster "
                  f"map but none was found; running on identity adjacency.")
    # run_ratio == 1.0 is the no-compression baseline (machinery stays cold);
    # run_ratio < 1.0 enables prefill-with-eviction KV compression.
    if run_ratio < 1.0:
        # Compression turns ON purely via compression_ratio < 1.0; there is no
        # separate enable flag on the LLM API / EngineArgs.
        llm_kwargs.update(
            compression_ratio=run_ratio,
            compression_chunk_size=args.compression_chunk_size,
            compression_n_sink_tokens=args.compression_n_sink_tokens,
            compression_window_size=args.compression_window_size,
            compression_floor_min=args.compression_floor_min,
            compression_gate_path=args.compression_gate_path,
            compression_scorer=args.compression_scorer,
            compression_level=args.compression_level,
            compression_snap_window=args.compression_snap_window,
            compression_snap_kernel=args.compression_snap_kernel,
            compression_ea_use_covariance=args.compression_ea_use_covariance,
            compression_ea_use_vnorm=args.compression_ea_use_vnorm,
            compression_ea_n_future_positions=(
                args.compression_ea_n_future_positions),
        )
        if args.compression_ea_epsilon is not None:
            llm_kwargs["compression_ea_epsilon"] = args.compression_ea_epsilon
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                        min_tokens=args.max_tokens, ignore_eos=True)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    e2e = time.perf_counter() - t0

    preemptions = 0
    for m in llm.get_metrics():
        if m.name == "vllm:num_preemptions":
            preemptions = int(getattr(m, "value", 0))
            break
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return {
        "ratio": run_ratio,
        "requests": len(prompts),
        "ctx_mean": sum(ctx_lens) // len(prompts),
        "preemptions": preemptions,
        "e2e_s": round(e2e, 1),
        "throughput_tok_s": round(gen_tokens / e2e, 1) if e2e > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Orchestrator: run baseline then compressed (each fresh) and print a table.
# ---------------------------------------------------------------------------
def _child_env() -> dict:
    env = dict(os.environ)
    # vLLM's engine core forks by default; CUDA already touched in the driver
    # makes the fork raise "Cannot re-initialize CUDA". Force spawn.
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # PYTORCH_ALLOC_CONF is the current name; older torch reads the CUDA-prefixed
    # one. Set whichever is unset so we stay quiet across versions.
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    return env


def run_config_in_subprocess(args: argparse.Namespace, run_ratio: float) -> dict:
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as f:
        out_path = f.name
    # Forward the *entire* parsed namespace (minus the per-run control fields) to
    # the worker via JSON, so every model/compression knob propagates without
    # enumerating flags here — adding a new arg never needs a change in this fn.
    shared = {k: v for k, v in vars(args).items()
              if k not in ("run_ratio", "_result_path", "_args_path")}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(shared, f)
        args_path = f.name
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--run-ratio", str(run_ratio),
        "--_args-path", args_path,
        "--_result-path", out_path,
    ]
    label = ("baseline (ratio 1.0, no compression)" if run_ratio >= 1.0
             else f"compressed (ratio {run_ratio})")
    print(f"\n>>> {label} ...", flush=True)
    proc = subprocess.run(cmd, env=_child_env())
    if proc.returncode != 0:
        os.unlink(args_path)
        raise SystemExit(f"ratio '{run_ratio}' run failed (exit {proc.returncode})")
    with open(out_path) as f:
        res = json.load(f)
    os.unlink(out_path)
    os.unlink(args_path)
    return res


def print_comparison(args: argparse.Namespace, base: dict, comp: dict) -> None:
    line = "─" * 70
    print("\n" + "═" * 70)
    print(" Tangram KV-cache compression — speedup benchmark")
    print("═" * 70)
    print(f" model      {os.path.basename(args.model.rstrip('/'))}")
    print(f" workload   {base['requests']} requests, ~{base['ctx_mean']} prompt tokens each")
    print(f" KV cache   gpu_memory_utilization={args.gpu_memory_utilization}")
    print(f" admission  scheduler_reserve_full_isl=ON (held constant)")
    print(line)
    print(f" {'configuration':<30}{'preemptions':>11}{'end-to-end':>13}{'throughput':>15}")
    print(f" {'baseline (ratio 1.0)':<30}{base['preemptions']:>11}"
          f"{str(base['e2e_s']) + ' s':>13}{str(base['throughput_tok_s']) + ' tok/s':>15}")
    print(f" {'compressed (ratio ' + str(comp['ratio']) + ')':<30}{comp['preemptions']:>11}"
          f"{str(comp['e2e_s']) + ' s':>13}{str(comp['throughput_tok_s']) + ' tok/s':>15}")
    print(f" compression scorer={args.compression_scorer} level={args.compression_level}")
    print(line)
    speedup = (base["e2e_s"] / comp["e2e_s"]) if comp["e2e_s"] else float("nan")
    print(f" → compression at ratio {comp['ratio']} removed "
          f"{base['preemptions'] - comp['preemptions']} preemptions and ran "
          f"{speedup:.2f}× faster than the uncompressed baseline")
    print("═" * 70 + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-m", "--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("-d", "--dataset", default="scbench_kv_short",
                   help="A long-context scbench dataset (default fits ~3-4 "
                        "requests at the default KV size).")
    p.add_argument("--num", type=int, default=24, help="Number of requests.")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                   help="Lower = tighter KV cache = more compression speedup.")
    p.add_argument("--max-tokens", type=int, default=128,
                   help="Output tokens per request (kept small to isolate the "
                        "prefill/KV-budget effect).")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--watermark", type=float, default=0.0,
                   help="Fraction of KV blocks kept free as decode headroom.")
    p.add_argument(
        "--run-ratio", type=float, default=None,
        help="Run a SINGLE config at this ratio instead of the comparison. "
             "1.0 = uncompressed baseline; < 1.0 = compressed.")
    _add_compression_args(p)
    # Internal: worker writes its result JSON here / reads shared args from here.
    p.add_argument("--_result-path", default=None, help=argparse.SUPPRESS)
    p.add_argument("--_args-path", default=None, help=argparse.SUPPRESS)
    return p


def _add_compression_args(p: argparse.ArgumentParser) -> None:
    """Tangram compression knobs. Defined inline (not imported) so the light
    orchestrator stays free of the heavy vLLM import."""
    g = p.add_argument_group("compression (prefill-with-eviction KV compression)")
    g.add_argument(
        "--ratio", type=float, default=0.3,
        help="KV cache budget as a ratio of the full cache, in (0, 1). This is "
             "the compressed configuration compared against the ratio-1.0 "
             "baseline. (The baseline ratio is always 1.0 and not configurable.)")
    g.add_argument("--page-group-size", type=int, default=4)
    g.add_argument(
        "--head-group-cluster-map", type=str, default=None,
        help="Path to a head-group cluster map .npz; None = identity grouping.")
    g.add_argument("--compression-gate-path", type=str, default="fastkvzip")
    g.add_argument("--compression-chunk-size", type=int, default=8192)
    g.add_argument("--compression-n-sink-tokens", type=int, default=32)
    g.add_argument("--compression-window-size", type=int, default=4096,
                   help="Recent tokens always kept during scoring.")
    g.add_argument("--compression-floor-min", type=int, default=512,
                   help="Per-(layer, group) kept-length floor in tokens.")
    g.add_argument(
        "--compression-scorer", type=str, default="snapkv",
        choices=["fastkvzip", "snapkv", "keydiff", "streamingllm", "tova",
                 "expected_attention"],
        help="Score producer (axis 2). All but 'fastkvzip' are gate-free.")
    g.add_argument(
        "--compression-level", default="perlayer_cluster",
        choices=("crosslayer_head", "perlayer_head", "crosslayer_cluster",
                 "perlayer_cluster", "uniform"),
        help="Selection level (axis 1), named {scope}_{granularity}.")
    g.add_argument("--compression-snap-window", type=int, default=32,
                   help="SnapKV observation window (trailing queries).")
    g.add_argument("--compression-snap-kernel", type=int, default=7,
                   help="SnapKV max-pool1d smoothing kernel size (odd).")
    g.add_argument("--compression-ea-use-covariance",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="ExpectedAttention: add the query-covariance term.")
    g.add_argument("--compression-ea-use-vnorm",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="ExpectedAttention: reweight by the value norm.")
    g.add_argument("--compression-ea-n-future-positions", type=int, default=512,
                   help="ExpectedAttention: #future positions averaged.")
    g.add_argument("--compression-ea-epsilon", type=float, default=None,
                   help="ExpectedAttention: constant before value-norm "
                        "reweighting; unset uses the engine default (1e-2).")


def main() -> None:
    args = build_parser().parse_args()

    # Worker mode: the orchestrator forwards the full shared namespace as JSON.
    # Overlay it so every model/compression knob matches the parent exactly.
    if args._args_path:
        with open(args._args_path) as f:
            for k, v in json.load(f).items():
                setattr(args, k, v)

    if not (0.0 < args.ratio < 1.0):
        raise SystemExit(
            f"--ratio must satisfy 0 < ratio < 1 (the compressed config), "
            f"got {args.ratio}. The baseline is always ratio 1.0.")

    # Single-config (worker) mode.
    if args.run_ratio is not None:
        if not (0.0 < args.run_ratio <= 1.0):
            raise SystemExit(
                f"--run-ratio must satisfy 0 < ratio <= 1, got {args.run_ratio}.")
        res = run_one(args, run_ratio=args.run_ratio)
        print(f"  result: {json.dumps(res)}")
        if args._result_path:
            with open(args._result_path, "w") as f:
                json.dump(res, f)
        return

    # Comparison mode: run baseline (1.0) then compressed (args.ratio) in fresh
    # processes, then tabulate the speedup.
    base = run_config_in_subprocess(args, 1.0)
    comp = run_config_in_subprocess(args, args.ratio)
    print_comparison(args, base, comp)


if __name__ == "__main__":
    main()
