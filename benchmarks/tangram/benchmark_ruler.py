# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline RULER benchmark for Tangram on vLLM.

Measures answer quality (and the same latency/throughput aggregates as the
SCBench driver) on RULER synthetic long-context tasks while sweeping the
FastKVZip compression ratio. RULER is single-turn: each sample is one
context+question prompt scored against its gold items with RULER's string-match
metric (recall for retrieval/tracking/extraction, any-match for QA).

The engine construction, compression knobs, and latency/throughput reporting are
shared with benchmark_scbench.py via ``bench_common``. RULER data loading, prompt
assembly, and scoring are provided self-contained by ``ruler_local`` (the
preprocessed ``simonjegou/ruler`` parquets; no external RULER checkout required).

Example:
    python benchmark_ruler.py --length 4096 --num 50 --ratio 0.3 \\
        -m Qwen/Qwen3-4B-Instruct-2507 --max-model-len 40960
"""

import argparse
import json
import os
import sys
import time
from typing import Any

from transformers import AutoTokenizer

# Generic engine + latency/throughput harness shared with benchmark_scbench.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_common import (  # noqa: E402
    add_compression_args,
    add_engine_args,
    build_benchmark_report,
    build_llm,
    extract_request_timing,
)
from ruler_local import (  # noqa: E402
    RULER_LENGTHS,
    RULER_TASKS,
    build_prompt,
    group_by_task,
    load_ruler,
    metric_name,
    score_answer,
)

from vllm import LLM, SamplingParams  # noqa: E402


# ---------------------------------------------------------------------------
# Per-task benchmark run
# ---------------------------------------------------------------------------

def run_task(
    *,
    llm: LLM,
    tokenizer: Any,
    model_basename: str,
    length: str,
    task: str,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Execute one (length, task, ratio) RULER run and return the result dict.

    Returns ``None`` when no sample survives the context-window fit check."""
    print("\n" + "=" * 70)
    print(f"  RULER {length} / {task}")
    print("=" * 70)

    # All samples of a task share its generation budget; --max-tokens overrides.
    max_tokens = args.max_tokens or samples[0]["max_new_tokens"]

    prompts: list[dict[str, list[int]]] = []
    answers: list[list[str]] = []
    input_lengths: list[int] = []
    skipped_too_long = 0
    for sample in samples:
        # Tokenize the chat-templated prompt ourselves (add_special_tokens=False)
        # so the templated special-token text is not re-doubled with a second
        # BOS -- matching benchmark_scbench's manual token assembly.
        prompt_ids = tokenizer.encode(
            build_prompt(tokenizer, model_basename, sample),
            add_special_tokens=False,
        )
        if args.max_model_len is not None and (
            len(prompt_ids) + max_tokens > args.max_model_len
        ):
            skipped_too_long += 1
            continue
        prompts.append({"prompt_token_ids": prompt_ids})
        answers.append(sample["answers"])
        input_lengths.append(len(prompt_ids))

    if skipped_too_long:
        print(
            f"  [length-skip] dropped {skipped_too_long} sample(s) exceeding "
            f"max_model_len={args.max_model_len}"
        )
    if not prompts:
        print("  No valid samples — skipping.")
        return None

    print(f"  Samples : {len(prompts)}  (metric: {metric_name(task)})")
    print(f"  Tokens  : max {max_tokens}/sample, "
          f"input median ~{sorted(input_lengths)[len(input_lengths) // 2]}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max_tokens,
        min_tokens=max_tokens if args.force_exact_tokens else 0,
        ignore_eos=bool(args.force_exact_tokens),
    )

    print("\n  Running generation ...")
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed_seconds = time.perf_counter() - start
    print(f"  Generation took {elapsed_seconds:.2f}s")

    scores: list[float] = []
    per_sample: list[dict[str, Any]] = []
    e2el_series: list[float] = []
    ttft_series: list[float] = []
    tpot_series: list[float] = []
    prefill_series: list[float] = []
    decode_series: list[float] = []
    queued_series: list[float] = []
    total_input_tokens = 0
    total_output_tokens = 0

    for idx, (output, gold, in_len) in enumerate(
        zip(outputs, answers, input_lengths)
    ):
        completion = output.outputs[0]
        prediction = completion.text
        out_len = len(completion.token_ids)
        score = score_answer(task, prediction, gold)
        scores.append(score)

        total_input_tokens += in_len
        total_output_tokens += out_len

        timing = extract_request_timing(getattr(output, "metrics", None))
        if timing["e2e_latency"] is not None:
            e2el_series.append(timing["e2e_latency"])
        if timing["ttft"] is not None:
            ttft_series.append(timing["ttft"])
        if timing["prefill_time"] is not None:
            prefill_series.append(timing["prefill_time"])
        if timing["decode_time"] is not None:
            decode_series.append(timing["decode_time"])
        if timing["queued_time"] is not None:
            queued_series.append(timing["queued_time"])
        if (
            timing["e2e_latency"] is not None
            and timing["ttft"] is not None
            and out_len > 1
        ):
            tpot_series.append(
                (timing["e2e_latency"] - timing["ttft"]) / (out_len - 1)
            )

        per_sample.append(
            {
                "sample_idx": idx,
                "input_tokens": in_len,
                "output_tokens": out_len,
                "score": float(score),
                "correct": bool(score >= 0.5),
                "prediction": prediction,
                "answers": gold,
                "timing_sec": timing,
            }
        )

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(
        f"\n  [{task}]  {len(scores)} samples  →  avg score: "
        f"{avg_score * 100:.2f}%"
    )
    for entry in per_sample[:3]:
        verdict = "correct" if entry["correct"] else "wrong"
        print(f"    [{entry['sample_idx']}] score={entry['score']:.2f} {verdict}")
        print(f"      pred: {entry['prediction'][:100]!r}")
        print(f"      gold: {entry['answers']}")

    benchmark = build_benchmark_report(
        elapsed_seconds=elapsed_seconds,
        num_conversations=len(per_sample),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        e2el_seconds=e2el_series,
        ttft_seconds=ttft_series,
        tpot_seconds=tpot_series,
        prefill_seconds=prefill_series,
        decode_seconds=decode_series,
        queued_seconds=queued_series,
        extra={
            "timing_source": "perf_counter+vllm.RequestOutput.metrics",
            "elapsed_kind": "single llm.generate() wall-clock",
        },
    )
    if benchmark["e2el_ms"]["count"]:
        print(
            f"  E2EL ms  mean={benchmark['e2el_ms']['mean']:.1f}  "
            f"p50={benchmark['e2el_ms']['percentiles']['p50']:.1f}  "
            f"p99={benchmark['e2el_ms']['percentiles']['p99']:.1f}  "
            f"|  TTFT mean={benchmark['ttft_ms']['mean']:.1f}ms  "
            f"|  out-tok/s={benchmark['output_throughput_tok_per_s']:.1f}"
        )

    return {
        "benchmark_suite": "ruler",
        "length": length,
        "task": task,
        "model": args.model_path,
        "compression_algo": args.compression_scorer,
        "compression_level": args.compression_level,
        # ExpectedAttention epsilon as requested; None = unset, so the engine's
        # CacheConfig.compression_ea_epsilon default (1e-2) applied.
        "compression_ea_epsilon": args.compression_ea_epsilon,
        "ratio": args.ratio,
        "page_group_size": args.page_group_size,
        "max_tokens": max_tokens,
        "metric": metric_name(task),
        "num_samples": len(per_sample),
        "avg_score": round(avg_score, 4),
        "generation_time_sec": round(elapsed_seconds, 4),
        "benchmark": benchmark,
        "scores": [float(s) for s in scores],
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """Define and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Offline RULER benchmark for Tangram on vLLM",
    )

    # Model / engine + compression (shared with benchmark_scbench.py).
    add_engine_args(parser)
    add_compression_args(parser)

    # Dataset (RULER-specific).
    parser.add_argument(
        "-l", "--length",
        type=str, default="4096", choices=list(RULER_LENGTHS),
        help="RULER context-length config to evaluate (one per run; the sweep "
             "script loops lengths). See ruler_local.RULER_LENGTHS.",
    )
    parser.add_argument(
        "--num", type=int, default=100,
        help="Samples to evaluate PER TASK (RULER ships 500/task).",
    )
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Comma-separated task filter (default: all). "
             f"Tasks: {', '.join(RULER_TASKS)}.",
    )
    parser.add_argument(
        "--force-exact-tokens", action="store_true", default=False,
        help="Force exactly max_tokens per sample (min_tokens=max, ignore_eos). "
             "Off by default — RULER scores natural-EOS generations.",
    )

    # Output.
    parser.add_argument(
        "--output-dir", type=str, default="./results_ruler",
        help="Directory to save evaluation results.",
    )
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Resume mode: skip any task whose result JSON already exists for "
             "this (length, ratio, page_group). If every task is already done, "
             "the model is not loaded. Lets an interrupted sweep be re-run with "
             "the same command to fill only the missing cells.",
    )

    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True,
    )
    model_basename = os.path.basename(args.model_path.rstrip("/"))
    tasks = (
        [t.strip() for t in args.tasks.split(",") if t.strip()]
        if args.tasks else None
    )

    tag_suffix = f"_{args.tag}" if args.tag else ""

    def save_path_for(task: str) -> str:
        return os.path.join(
            args.output_dir, f"len{args.length}", task,
            f"{model_basename}_r{args.ratio}_pg{args.page_group_size}"
            f"{tag_suffix}.json",
        )

    dataset = load_ruler(args.length, n_data=args.num, tasks=tasks)
    grouped = group_by_task(dataset)
    task_names = [t for t in RULER_TASKS if t in grouped]
    # Include any task present in the data but not in the canonical list.
    task_names += [t for t in grouped if t not in task_names]

    # Resume mode: drop tasks already computed for this (length, ratio,
    # page_group); skip the model load entirely when nothing is pending.
    if args.skip_existing:
        done = [t for t in task_names if os.path.exists(save_path_for(t))]
        if done:
            print(f"[resume] length={args.length} ratio={args.ratio}: skipping "
                  f"{len(done)} existing ({', '.join(done)})")
        task_names = [t for t in task_names if not os.path.exists(save_path_for(t))]
        if not task_names:
            print(f"[resume] length={args.length} ratio={args.ratio}: all tasks "
                  "done; skipping model load.")
            return

    llm = build_llm(args)

    for task in task_names:
        result = run_task(
            llm=llm,
            tokenizer=tokenizer,
            model_basename=model_basename,
            length=args.length,
            task=task,
            samples=grouped[task],
            args=args,
        )
        if result is None:
            continue

        save_path = save_path_for(task)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Results saved → {save_path}")
        print("-" * 70)

    print("\n" + "=" * 70)
    print("  All RULER evaluations finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()
