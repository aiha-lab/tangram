# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared harness for the Tangram offline accuracy benchmarks.

Both ``benchmark_scbench.py`` (SCBench) and ``benchmark_ruler.py`` (RULER) drive
the same vLLM compression engine and report the same latency/throughput shape;
only the dataset loading, prompt assembly, and scoring differ. The generic parts
live here so the two drivers stay thin and identical where they should be:

  * latency/throughput aggregation — ``summarize`` / ``extract_request_timing``
    / ``build_benchmark_report``
  * engine construction with the Tangram compression knobs — ``build_llm``
  * the CLI argument groups those share — ``add_engine_args`` /
    ``add_compression_args``

A driver builds its parser by calling the two ``add_*_args`` helpers, then adds
its own dataset/output arguments, and passes the parsed namespace to
``build_llm``. ``build_llm`` reads only attributes set by these shared groups, so
any driver that includes both groups can construct the engine unchanged.
"""

import argparse
import os
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from vllm import LLM
from vllm.v1.attention.compression.scorer_options import (
    parse_scorer_options,
)


DEFAULT_PERCENTILES: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0)


# ---------------------------------------------------------------------------
# Latency / throughput aggregation
# ---------------------------------------------------------------------------

def summarize(
    values: Sequence[float],
    percentiles: Iterable[float] = DEFAULT_PERCENTILES,
    scale: float = 1000.0,
) -> dict[str, Any]:
    """Aggregate a series in seconds to mean/median/std/min/max/percentiles in
    milliseconds. Empty input yields all zeros."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "percentiles": {f"p{int(p)}": 0.0 for p in percentiles},
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean() * scale),
        "median": float(np.median(arr) * scale),
        "std": float(arr.std() * scale),
        "min": float(arr.min() * scale),
        "max": float(arr.max() * scale),
        "percentiles": {
            f"p{int(p)}": float(np.percentile(arr, p) * scale)
            for p in percentiles
        },
    }


def extract_request_timing(metrics_obj: Any) -> dict[str, float | None]:
    """Read per-request latencies (seconds) from a ``RequestOutput.metrics``.
    Definitions mirror vllm/v1/metrics/loggers.py: queued = scheduled - queued,
    prefill = first_token - scheduled, decode = last_token - first_token,
    e2e = last_token - arrival, ttft = first_token_latency."""
    if metrics_obj is None:
        return {
            key: None
            for key in (
                "arrival_time",
                "queued_time",
                "prefill_time",
                "decode_time",
                "inference_time",
                "e2e_latency",
                "ttft",
                "num_generation_tokens",
            )
        }

    arrival_time = getattr(metrics_obj, "arrival_time", None)
    queued_ts = getattr(metrics_obj, "queued_ts", None)
    scheduled_ts = getattr(metrics_obj, "scheduled_ts", None)
    first_token_ts = getattr(metrics_obj, "first_token_ts", None)
    last_token_ts = getattr(metrics_obj, "last_token_ts", None)
    ttft = getattr(metrics_obj, "first_token_latency", None)
    num_generation_tokens = getattr(metrics_obj, "num_generation_tokens", None)

    def difference(a: float | None, b: float | None) -> float | None:
        if a is None or b is None or a == 0.0 or b == 0.0:
            return None
        return float(a - b)

    queued_time = difference(scheduled_ts, queued_ts)
    prefill_time = difference(first_token_ts, scheduled_ts)
    decode_time = difference(last_token_ts, first_token_ts)
    inference_time = difference(last_token_ts, scheduled_ts)

    # arrival_time is wall-clock and *_ts fields are monotonic; the subtraction
    # may go negative on wall-clock skew. Fall back to ttft + decode_time then.
    e2e_latency: float | None = None
    if last_token_ts and arrival_time:
        candidate = float(last_token_ts - arrival_time)
        e2e_latency = candidate if candidate > 0 else None
    if e2e_latency is None and ttft is not None and decode_time is not None:
        e2e_latency = float(ttft + decode_time)

    return {
        "arrival_time": float(arrival_time) if arrival_time else None,
        "queued_time": queued_time,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "inference_time": inference_time,
        "e2e_latency": e2e_latency,
        "ttft": float(ttft) if ttft else None,
        "num_generation_tokens": (
            int(num_generation_tokens)
            if num_generation_tokens is not None
            else None
        ),
    }


def build_benchmark_report(
    *,
    elapsed_seconds: float,
    num_conversations: int,
    total_input_tokens: int,
    total_output_tokens: int,
    e2el_seconds: Sequence[float],
    ttft_seconds: Sequence[float],
    tpot_seconds: Sequence[float],
    prefill_seconds: Sequence[float],
    decode_seconds: Sequence[float],
    queued_seconds: Sequence[float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a vLLM ``BenchmarkMetrics``-shaped report for JSON dump."""
    request_throughput = (
        num_conversations / elapsed_seconds if elapsed_seconds > 0 else 0.0
    )
    output_throughput = (
        total_output_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
    )
    total_throughput = (
        (total_input_tokens + total_output_tokens) / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0
    )
    report: dict[str, Any] = {
        "elapsed_sec": round(elapsed_seconds, 4),
        "num_conversations": num_conversations,
        "total_input_tokens": int(total_input_tokens),
        "total_output_tokens": int(total_output_tokens),
        "request_throughput_per_s": round(request_throughput, 4),
        "output_throughput_tok_per_s": round(output_throughput, 4),
        "total_token_throughput_tok_per_s": round(total_throughput, 4),
        "e2el_ms": summarize(e2el_seconds),
        "ttft_ms": summarize(ttft_seconds),
        "tpot_ms": summarize(tpot_seconds),
        "prefill_ms": summarize(prefill_seconds),
        "decode_ms": summarize(decode_seconds),
        "queued_ms": summarize(queued_seconds),
    }
    if extra:
        report.update(extra)
    return report


# ---------------------------------------------------------------------------
# Shared CLI argument groups
# ---------------------------------------------------------------------------

def add_engine_args(parser: argparse.ArgumentParser) -> None:
    """Add the model / engine arguments shared by every accuracy driver.

    Defaults match the historical ``benchmark_scbench.py`` so its CLI is
    unchanged; RULER overrides only its own dataset/output arguments."""
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-1M",
        help="Model checkpoint path or HF model ID.",
    )
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Max output tokens per turn; auto-set per dataset when omitted.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--tensor-parallel-size", "--tp",
        type=int, default=1, dest="tensor_parallel_size",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--max-num-seqs", type=int, default=None,
        help=(
            "Cap on concurrently-running sequences. Set to 1 to fully "
            "serialize execution (no batching, no preemption)."
        ),
    )
    parser.add_argument(
        "--num-gpu-blocks-override", type=int, default=None,
        help=(
            "Fix the KV cache at this many blocks instead of profiling for it. "
            "A pool too small to hold every admitted request forces the "
            "scheduler to preempt, which is otherwise hard to reach under "
            "compression because a budget caps each request's KV."
        ),
    )
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=None,
        help=(
            "Cap on prefill tokens batched into one forward step (vLLM "
            "chunked-prefill knob). Leave unset for vLLM's default. Set it "
            ">= the input length to force a SINGLE-SHOT prefill (no sub-"
            "chunking), so the scorer sees the whole context at once -- "
            "needed to match the kvpress one-shot reference for "
            "ExpectedAttention."
        ),
    )
    parser.add_argument(
        "--attention-backend", type=str, default="FLASH_ATTN",
        choices=["FLASH_ATTN", "FLASHINFER"],
        help="Sets VLLM_ATTENTION_BACKEND for the engine.",
    )
    parser.add_argument(
        "--enable-prefix-caching", action="store_true", default=False,
        help="Enable vLLM prefix caching (off by default for clean numbers).",
    )
    parser.add_argument(
        "--enable-log-stats", action="store_true", default=False,
        help=(
            "Enable vLLM's periodic engine stats logger (throughput, queue "
            "depths, preemption count). Off by default to keep output clean."
        ),
    )
    parser.add_argument(
        "--disable-custom-all-reduce", action="store_true", default=False,
    )


def effective_ratio(args: argparse.Namespace) -> float | None:
    """The evicted fraction the engine runs with (``compression_ratio``);
    ``None`` when a budget is the retention target or no ratio is set."""
    if args.compression_budget_tokens is not None or not args.compression_ratio:
        return None
    return args.compression_ratio


def add_compression_args(parser: argparse.ArgumentParser) -> None:
    """Add the Tangram compression (FastKVZip prefill-with-eviction) arguments
    shared by every accuracy driver. ``build_llm`` consumes exactly these."""
    parser.add_argument(
        "--compression-ratio", type=float, default=0.0,
        help=(
            "Fraction of the KV cache to evict, in [0, 1). "
            "0 (default) disables compression and runs the baseline."
        ),
    )
    parser.add_argument(
        "--compression-budget-tokens", type=int, default=None,
        help=(
            "Fixed KV cache budget in tokens per (layer, head group). "
            "Mutually exclusive with --compression-ratio: setting it selects the "
            "budget eviction regime, where nothing is evicted until the cache "
            "would exceed the budget and it is then cut back to it. "
            "--compression-ratio is ignored when a budget is given."
        ),
    )
    parser.add_argument(
        "--compression-evict-current-chunk", action="store_true",
        help=(
            "Budget regime only: let the chunk just written compete for "
            "eviction, protecting only the always-kept recent window. Off by "
            "default, which protects the whole fresh chunk and is the more "
            "accurate setting in our measurements."
        ),
    )
    parser.add_argument(
        "--compression-slot-score-source", type=str, default="auto",
        choices=("auto", "persist", "recompute"),
        help=(
            "Budget regime only: where a cached position's score comes from "
            "when it competes again. 'auto' (default) takes what the scorer "
            "specifies. Forcing 'persist' with a rescoring scorer (keydiff) is "
            "the ablation that separates the retention target from the score: "
            "it ranks the chunk-local scores a ratio run also ranks, so what "
            "remains between a ratio and a budget run is the target alone."
        ),
    )
    parser.add_argument(
        "--compression-scorer-options", type=str, default="",
        help=(
            "Settings the selected --compression-scorer declares, as "
            "key=value,key=value (see the scorer's OPTIONS). Example: "
            "--compression-scorer keydiff --compression-scorer-options "
            "anchor=normalized to rank by KeyDiff Eq. (8)'s normalized anchor "
            "instead of the unnormalized mean its experiments use."
        ),
    )
    parser.add_argument("--page-group-size", type=int, default=4)
    parser.add_argument(
        "--head-group-cluster-map", type=str, default=None,
        help=(
            "Path to a head-group cluster map .npz (see "
            "tools/head_group_clustering). None = identity (adjacent-head) "
            "grouping. A cross-layer map groups similar-budget heads so the "
            "per-cluster max-pooled kept length approaches each member's need, "
            "reducing KV memory."
        ),
    )
    parser.add_argument("--compression-gate-path", type=str, default="fastkvzip")
    parser.add_argument("--compression-chunk-size", type=int, default=8192)
    parser.add_argument("--compression-n-sink-tokens", type=int, default=32)
    parser.add_argument(
        "--compression-window-size", type=int, default=4096,
        help="Recent tokens always kept during scoring.",
    )
    parser.add_argument(
        "--compression-floor-min", type=int, default=512,
        help="Per-(layer, group) kept-length floor in tokens; 0 disables.",
    )
    # Two orthogonal axes (budget scope + scorer).
    parser.add_argument(
        "--compression-scorer", type=str, default="fastkvzip",
        choices=["fastkvzip", "snapkv", "keydiff", "streamingllm", "tova",
                 "expected_attention"],
        help="Score producer (axis 2). All but 'fastkvzip' are gate-free (no "
             "checkpoint); scores come from the model's post-RoPE query/key "
             "per chunk (SnapKV: observation-window attention; KeyDiff: key "
             "similarity to the chunk mean key; StreamingLLM: token recency / "
             "global position; TOVA: last-query attention averaged across "
             "heads; ExpectedAttention: analytic expected attention of future "
             "queries with optional covariance + value-norm).",
    )
    parser.add_argument(
        "--compression-budget-scope", default="layer",
        choices=("uniform", "layer", "global"),
        help="Scope the retention budget is balanced over (axis 1). "
             "'layer' (default): each layer gets an equal budget, pooled "
             "non-uniformly across its head groups (needs a per-layer cluster "
             "map). 'global': one budget pooled across all layers and head "
             "groups (needs a global cluster map). 'uniform': every "
             "(layer, group) keeps the same count.",
    )


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------

def build_llm(args: argparse.Namespace) -> LLM:
    """Construct the vLLM engine with Tangram options derived from ``args``.

    Reads only attributes added by ``add_engine_args`` / ``add_compression_args``
    (plus optional ``args.multi_turn``), so any driver including both groups can
    call this unchanged."""
    if not (0.0 <= args.compression_ratio < 1.0):
        raise ValueError(
            "--compression-ratio is the fraction of the KV cache to evict "
            "and must "
            f"satisfy 0 <= ratio < 1, got {args.compression_ratio}."
        )

    # LLM defaults disable_log_stats=True, leaving RequestOutput.metrics None;
    # we derive throughput from elapsed time + token counts. --enable-log-stats
    # turns the metrics back on.
    llm_kwargs: dict[str, Any] = {
        "model": args.model_path,
        "dtype": "auto",
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        # TANGRAM_GRAPH=1 runs the engine in its compiled mode (VLLM_COMPILE
        # + CUDA graphs; ragged runs are pinned to PIECEWISE at config
        # time). Default stays eager so existing sweep results remain
        # directly comparable.
        "enforce_eager": os.environ.get("TANGRAM_GRAPH", "0") != "1",
        "max_model_len": args.max_model_len,
        "enable_prefix_caching": args.enable_prefix_caching,
        "disable_log_stats": not args.enable_log_stats,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "page_group_size": args.page_group_size,
        "head_group_cluster_map": args.head_group_cluster_map,
        # Multi-turn auto-advance is needed only by the SCBench driver; RULER is
        # single-turn. The engine flag is harmless when no multi-turn token IDs
        # are passed to generate(), so it defaults on for backward-compatibility.
        "multi_turn": getattr(args, "multi_turn", True),
    }
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.num_gpu_blocks_override is not None:
        llm_kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks_override
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    # Either retention target turns compression on; with neither, the machinery
    # stays cold so we get a true reference point against the swept settings.
    budget_tokens = args.compression_budget_tokens
    ratio = effective_ratio(args)
    if budget_tokens is not None and args.compression_ratio:
        print(f"  [budget] ignoring --compression-ratio "
              f"{args.compression_ratio}: "
              f"--compression-budget-tokens {budget_tokens} is the "
              "retention target.")
    if ratio is not None or budget_tokens is not None:
        llm_kwargs.update(
            compression_ratio=ratio,
            compression_budget_tokens=budget_tokens,
            compression_evict_current_chunk=args.compression_evict_current_chunk,
            compression_slot_score_source=args.compression_slot_score_source,
            compression_scorer_options=parse_scorer_options(
                args.compression_scorer_options or ""),
            compression_chunk_size=args.compression_chunk_size,
            compression_n_sink_tokens=args.compression_n_sink_tokens,
            compression_window_size=args.compression_window_size,
            compression_floor_min=args.compression_floor_min,
            compression_gate_path=args.compression_gate_path,
            compression_scorer=args.compression_scorer,
            compression_budget_scope=args.compression_budget_scope,
        )

    print(f"\nLoading model from {args.model_path} ...")
    return LLM(**llm_kwargs)
