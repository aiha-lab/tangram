#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark: KV-cache admission control vs. preemption thrashing.

By default, under chunked prefill the scheduler admits a request as soon as its
*first chunk* fits in the KV cache — not its full length. When the cache is
tight this over-admits: admitted requests keep growing, run out of cache, and
get **preempted** (their KV is discarded and recomputed from scratch). Under
load this preemption thrashing can collapse throughput.

The ``scheduler_reserve_full_isl`` engine option makes the scheduler admit a
request only if its *entire* input length fits, which prevents the
over-admission. This script runs the same KV-pressured workload with that option
OFF then ON and prints a side-by-side comparison of preemptions and throughput.

Tangram KV-cache compression (FastKVZip prefill-with-eviction and the gate-free
scorers) can be layered on top via ``--ratio`` (< 1.0 enables compression) plus
the ``--compression-*`` knobs, which mirror ``bench_common.add_compression_args``
one-for-one so the engine is built with the same semantics as the accuracy
drivers. The admission OFF-vs-ON comparison then runs *under* compression — note
that today's full-ISL reserve still sizes the reservation off the *uncompressed*
``num_tokens`` (compression-aware admission is track item 02, not yet built), so
``ON`` will be over-conservative under compression; that gap is exactly what this
harness exists to quantify.

Examples
--------
    # Run the OFF-vs-ON comparison (default). Pin a free GPU as usual:
    CUDA_VISIBLE_DEVICES=0 python benchmark.py

    # Squeeze the KV cache harder (fewer requests fit -> more thrashing):
    CUDA_VISIBLE_DEVICES=0 python benchmark.py --gpu-memory-utilization 0.12 --num 32

    # Run a single configuration only:
    CUDA_VISIBLE_DEVICES=0 python benchmark.py --reserve off

    # Same comparison, but with FastKVZip compression at a 30% KV budget:
    CUDA_VISIBLE_DEVICES=0 python benchmark.py --ratio 0.3

    # A gate-free scorer (no checkpoint) with a per-layer selection level:
    CUDA_VISIBLE_DEVICES=0 python benchmark.py \\
        --ratio 0.3 --compression-scorer snapkv --compression-level perlayer_head
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


# ---------------------------------------------------------------------------
# Worker: run ONE configuration in this process and report its metrics.
# ---------------------------------------------------------------------------
def run_one(args: argparse.Namespace, reserve_full_isl: bool) -> dict:
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
        # Compression forces prefix caching off (cache.py); keep it off in the
        # baseline too so OFF/ON differ only in admission control.
        enable_prefix_caching=False,
        disable_log_stats=False,  # required so get_metrics() is populated
        scheduler_reserve_full_isl=reserve_full_isl,
        watermark=args.watermark,
        # Head-grouped paging layout — always on in tangram-asp (pg=4 default).
        page_group_size=args.page_group_size,
        head_group_cluster_map=args.head_group_cluster_map,
    )
    # ratio == 1.0 is the no-compression baseline; the compression machinery
    # stays cold. ratio < 1.0 enables FastKVZip prefill-with-eviction. These
    # knobs mirror bench_common.build_llm exactly.
    if args.ratio < 1.0:
        llm_kwargs.update(
            enable_compression=True,
            compression_ratio=args.ratio,
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
        "reserve_full_isl": reserve_full_isl,
        "requests": len(prompts),
        "ctx_mean": sum(ctx_lens) // len(prompts),
        "preemptions": preemptions,
        "e2e_s": round(e2e, 1),
        "throughput_tok_s": round(gen_tokens / e2e, 1) if e2e > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Orchestrator: run OFF then ON (each in a fresh process) and print a table.
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


def run_config_in_subprocess(args: argparse.Namespace, reserve: str) -> dict:
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as f:
        out_path = f.name
    # Forward the *entire* parsed namespace (minus the per-run control fields) to
    # the worker via JSON, so every model/compression knob propagates without
    # enumerating flags here — adding a new arg never needs a change in this fn.
    shared = {k: v for k, v in vars(args).items()
              if k not in ("reserve", "_result_path", "_args_path")}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(shared, f)
        args_path = f.name
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--reserve", reserve,
        "--_args-path", args_path,
        "--_result-path", out_path,
    ]
    label = "OFF (greedy admission)" if reserve == "off" else "ON  (reserve full input length)"
    print(f"\n>>> admission control {label} ...", flush=True)
    proc = subprocess.run(cmd, env=_child_env())
    if proc.returncode != 0:
        os.unlink(args_path)
        raise SystemExit(f"configuration '{reserve}' failed (exit {proc.returncode})")
    with open(out_path) as f:
        res = json.load(f)
    os.unlink(out_path)
    os.unlink(args_path)
    return res


def print_comparison(args: argparse.Namespace, off: dict, on: dict) -> None:
    line = "─" * 70
    print("\n" + "═" * 70)
    print(" KV-cache admission control — preemption benchmark")
    print("═" * 70)
    print(f" model      {os.path.basename(args.model.rstrip('/'))}")
    print(f" workload   {off['requests']} requests, ~{off['ctx_mean']} prompt tokens each")
    print(f" KV cache   gpu_memory_utilization={args.gpu_memory_utilization} (intentionally tight)")
    print(line)
    print(f" {'admission control':<26}{'preemptions':>13}{'end-to-end':>13}{'throughput':>15}")
    print(f" {'OFF (greedy, default)':<26}{off['preemptions']:>13}"
          f"{str(off['e2e_s']) + ' s':>13}{str(off['throughput_tok_s']) + ' tok/s':>15}")
    print(f" {'ON  (full-input reserve)':<26}{on['preemptions']:>13}"
          f"{str(on['e2e_s']) + ' s':>13}{str(on['throughput_tok_s']) + ' tok/s':>15}")
    if args.ratio < 1.0:
        print(f" compression ratio={args.ratio} scorer={args.compression_scorer} "
              f"level={args.compression_level}")
    else:
        print(" compression OFF (ratio=1.0, baseline)")
    print(line)
    speedup = (off["e2e_s"] / on["e2e_s"]) if on["e2e_s"] else float("nan")
    print(f" → admission control removed {off['preemptions'] - on['preemptions']} "
          f"preemptions and ran {speedup:.2f}× faster")
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
                   help="Lower = tighter KV cache = more preemption pressure.")
    p.add_argument("--max-tokens", type=int, default=128,
                   help="Output tokens per request (kept small to isolate the "
                        "prefill-admission effect).")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--watermark", type=float, default=0.0,
                   help="Fraction of KV blocks kept free as decode headroom.")
    p.add_argument("--reserve", choices=["on", "off"], default=None,
                   help="Run a SINGLE config instead of the comparison.")
    _add_compression_args(p)
    # Internal: worker writes its result JSON here / reads shared args from here.
    p.add_argument("--_result-path", default=None, help=argparse.SUPPRESS)
    p.add_argument("--_args-path", default=None, help=argparse.SUPPRESS)
    return p


def _add_compression_args(p: argparse.ArgumentParser) -> None:
    """Tangram compression knobs, mirroring bench_common.add_compression_args
    one-for-one so the engine is built with identical semantics. Defined inline
    (not imported) to keep the light orchestrator free of the heavy vLLM import
    that bench_common performs at module load."""
    g = p.add_argument_group("compression (FastKVZip prefill-with-eviction)")
    g.add_argument(
        "--ratio", type=float, default=1.0,
        help="KV cache budget as a ratio of the full cache, in (0, 1]. "
             "ratio == 1.0 (default) disables compression and runs the "
             "uncompressed baseline; < 1.0 enables compression.")
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
        "--compression-level", default="crosslayer_head",
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

    if not (0.0 < args.ratio <= 1.0):
        raise SystemExit(f"--ratio must satisfy 0 < ratio <= 1, got {args.ratio}.")

    # Single-config (worker) mode.
    if args.reserve is not None:
        res = run_one(args, reserve_full_isl=(args.reserve == "on"))
        print(f"  result: {json.dumps(res)}")
        if args._result_path:
            with open(args._result_path, "w") as f:
                json.dump(res, f)
        return

    # Comparison mode: run OFF then ON in fresh processes, then tabulate.
    off = run_config_in_subprocess(args, "off")
    on = run_config_in_subprocess(args, "on")
    print_comparison(args, off, on)


if __name__ == "__main__":
    main()
