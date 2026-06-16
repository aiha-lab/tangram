# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline SCBench benchmark for Tangram on vLLM.

Measures throughput, per-request latency, and answer quality on SCBench
datasets while sweeping the FastKVZip compression ratio. Each turn carries
a single conversation through vLLM's multi-turn auto-advance path; turn 0
is a prefill-only context turn (one output token, discarded), turn 1 emits
the answer used for evaluation.

SCBench data loading, per-model prompt templates, generation-length presets,
and answer metrics are provided self-contained by ``scbench_local`` (a sibling
module in this directory), so no external FastKVZip checkout is required.

Example:
    python benchmark_scbench.py -d scbench_kv --num 100 --ratio 0.3 \\
        -m Qwen/Qwen2.5-7B-Instruct-1M --max-model-len 200000 \\
        --single-turn --force-exact-tokens --max-tokens 512
"""

import argparse
import json
import os
import sys
import time
from typing import Any

from transformers import AutoTokenizer

# Self-contained SCBench helpers (formerly FastKVZip's prefill package). Ensure
# this script's directory is importable when invoked from elsewhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Generic engine + latency/throughput harness shared with benchmark_ruler.py.
from bench_common import (  # noqa: E402
    add_compression_args,
    add_engine_args,
    build_benchmark_report,
    build_llm,
    extract_request_timing,
    summarize,
)
from scbench_local import (  # noqa: E402
    evaluate_answer,
    f1_score,
    get_data_list,
    load_dataset_all,
    set_gen_length,
    template,
)

from vllm import LLM, SamplingParams  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset → multi-turn conversion
# ---------------------------------------------------------------------------

def get_query_text(task: str, question: str) -> str:
    """Wrap a question with the task-specific instruction prefix."""
    if task == "reason":
        return (
            "Reason and answer the question. You must say the answer in the "
            f"last sentence beginning with 'The answer is'. Q: {question}"
        )
    return f"Q: {question}"


def get_eval_task(dataset_name: str) -> str:
    """Map a dataset name to its evaluation task type."""
    if "gsm" in dataset_name:
        return "reason"
    return "qa"


def build_multi_turn_from_dataset(
    *,
    tokenizer: Any,
    model_name: str,
    dataset: Any,
    dataset_name: str,
    start_idx: int,
    end_idx: int,
    max_questions: int | None,
    max_tokens: int,
    max_model_len: int | None = None,
) -> tuple[
    list[dict[str, list[int]]],
    list[list[list[int]]],
    list[list[int]],
    list[list[str]],
    list[int],
]:
    """Convert dataset samples into per-conversation turn token IDs.

    Token assembly mirrors FastKVZip's DataWrapper / ModelKVzip exactly:
      Turn 0 = encode(prefix) + encode(context)        # prefill-only
      Turn k = encode("\\n\\n{query}") + encode(postfix)   # generation
    vLLM uses ``multi_turn_token_ids[i][0]`` as the actual turn-0 prompt
    (overriding ``prompts``), so we also pass turn-0 IDs as the prompt.
    """
    task = get_eval_task(dataset_name)
    prefix, postfix = template(model_name, dataset_name)

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    prefix_ids = encode(prefix)
    postfix_ids = encode(postfix)

    prompts: list[dict[str, list[int]]] = []
    multi_turn_token_ids: list[list[list[int]]] = []
    turn_max_tokens: list[list[int]] = []
    ground_truths: list[list[str]] = []
    sample_indices: list[int] = []
    skipped_too_long = 0

    for data_idx in range(start_idx, end_idx):
        sample = dataset[data_idx]
        context = sample["context"]
        questions = list(sample["question"])
        answers = list(sample["answers"])

        if max_questions is not None:
            questions = questions[:max_questions]
            answers = answers[:max_questions]

        if not questions or not questions[0]:
            continue

        context_ids = encode(context)
        turn0_ids = prefix_ids + context_ids

        per_conv_turns = [turn0_ids]
        for question in questions:
            query_text = get_query_text(task, question)
            per_conv_turns.append(
                encode(f"\n\n{query_text.strip()}") + postfix_ids
            )

        # Turn 0 emits one throwaway token (context-only prefill); turns 1+ run
        # full generation up to ``max_tokens``.
        per_conv_max_tokens = [1] + [max_tokens] * len(questions)

        # Safety net: drop a sample whose full conversation (all turn prompts +
        # their generations) cannot fit the context window. Tokenizer-dependent
        # lengths (e.g. full scbench_kv ~120-170k) can still exceed max_model_len
        # after capacity-based variant selection; skip+count instead of crashing
        # the whole run on vLLM's prompt-length validation.
        if max_model_len is not None:
            conv_len = sum(len(t) for t in per_conv_turns) + max_tokens * len(questions)
            if conv_len > max_model_len:
                skipped_too_long += 1
                continue

        prompts.append({"prompt_token_ids": turn0_ids})
        multi_turn_token_ids.append(per_conv_turns)
        turn_max_tokens.append(per_conv_max_tokens)
        ground_truths.append(answers)
        sample_indices.append(data_idx)

    if skipped_too_long:
        print(
            f"  [length-skip] dropped {skipped_too_long} sample(s) exceeding "
            f"max_model_len={max_model_len} for {dataset_name}"
        )

    return (
        prompts,
        multi_turn_token_ids,
        turn_max_tokens,
        ground_truths,
        sample_indices,
    )


# ---------------------------------------------------------------------------
# Per-dataset benchmark run
# ---------------------------------------------------------------------------

def run_dataset(
    *,
    llm: LLM,
    tokenizer: Any,
    model_basename: str,
    dataset_name: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Execute one (dataset, ratio) benchmark run and return the result dict.

    Returns ``None`` when the dataset slice yields no usable samples."""
    print("\n" + "=" * 70)
    print(f"  Dataset: {dataset_name}")
    print("=" * 70)

    dataset = load_dataset_all(dataset_name, tokenizer, n_data=args.num)
    end_idx = min(args.idx + args.num, len(dataset))
    max_tokens = args.max_tokens or set_gen_length(dataset_name)
    max_questions = 1 if args.single_turn else args.max_questions
    eval_task = get_eval_task(dataset_name)

    (
        prompts,
        multi_turn_token_ids,
        turn_max_tokens,
        ground_truths,
        sample_indices,
    ) = build_multi_turn_from_dataset(
        tokenizer=tokenizer,
        model_name=model_basename,
        dataset=dataset,
        dataset_name=dataset_name,
        start_idx=args.idx,
        end_idx=end_idx,
        max_questions=max_questions,
        max_tokens=max_tokens,
        max_model_len=args.max_model_len,
    )

    if not prompts:
        print("  No valid samples — skipping.")
        return None

    print(
        f"  Samples : {args.idx} ~ {end_idx}  "
        f"({len(prompts)} conversations)"
    )
    print(f"  Task    : {eval_task}")
    print(f"  Tokens  : max {max_tokens}/turn")
    for sample_idx, turns in zip(sample_indices[:5], multi_turn_token_ids[:5]):
        turn_lengths = ", ".join(
            f"T{turn_idx}={len(ids)}" for turn_idx, ids in enumerate(turns)
        )
        print(f"    [{sample_idx}] {len(turns)} turns  ({turn_lengths})")
    if len(prompts) > 5:
        print(f"    ... and {len(prompts) - 5} more")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max_tokens,
        min_tokens=max_tokens if args.force_exact_tokens else 0,
        ignore_eos=bool(args.force_exact_tokens),
    )

    print("\n  Running multi-turn generation ...")
    start = time.perf_counter()
    outputs = llm.generate(
        prompts,
        sampling_params,
        multi_turn_token_ids=multi_turn_token_ids,
        turn_max_tokens=turn_max_tokens,
    )
    elapsed_seconds = time.perf_counter() - start
    print(f"  Generation took {elapsed_seconds:.2f}s")

    predictions_flat: list[str] = []
    references_flat: list[str] = []
    per_sample: list[dict[str, Any]] = []
    e2el_series: list[float] = []
    ttft_series: list[float] = []
    tpot_series: list[float] = []
    prefill_series: list[float] = []
    decode_series: list[float] = []
    queued_series: list[float] = []
    total_input_tokens = 0
    total_output_tokens = 0

    for conv_idx, (output, gt_answers, sample_idx) in enumerate(
        zip(outputs, ground_truths, sample_indices)
    ):
        sample_predictions: list[str] = []
        sample_output_tokens = 0

        # turn_output_token_ids exists in the multi-turn path; the first entry
        # is turn 0 (context-only) and is dropped before evaluation.
        if output.turn_output_token_ids:
            for turn_ids in output.turn_output_token_ids[1:]:
                sample_predictions.append(
                    tokenizer.decode(turn_ids, skip_special_tokens=True)
                )
                sample_output_tokens += len(turn_ids)
        else:
            completion = output.outputs[0]
            sample_predictions.append(completion.text)
            sample_output_tokens += len(completion.token_ids)

        n_eval = min(len(sample_predictions), len(gt_answers))
        predictions_flat.extend(sample_predictions[:n_eval])
        references_flat.extend(gt_answers[:n_eval])

        per_turn_input_lengths = [
            len(ids) for ids in multi_turn_token_ids[conv_idx]
        ]
        sample_input_tokens = sum(per_turn_input_lengths)
        total_input_tokens += sample_input_tokens
        total_output_tokens += sample_output_tokens

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
        # TPOT formula (matches vLLM serve): (e2e - ttft) / (n_gen - 1).
        if (
            timing["e2e_latency"] is not None
            and timing["ttft"] is not None
            and sample_output_tokens > 1
        ):
            tpot_series.append(
                (timing["e2e_latency"] - timing["ttft"])
                / (sample_output_tokens - 1)
            )

        per_sample.append(
            {
                "sample_idx": sample_idx,
                "num_turns": len(multi_turn_token_ids[conv_idx]),
                "input_token_lengths": per_turn_input_lengths,
                "total_input_tokens": sample_input_tokens,
                "total_output_tokens": sample_output_tokens,
                "predictions": sample_predictions[:n_eval],
                "ground_truths": gt_answers[:n_eval],
                "timing_sec": timing,
            }
        )

    avg_score = 0.0
    scores: list[float] = []
    if predictions_flat and references_flat:
        # repoqa carries a structured refs payload that the non-similarity path
        # cannot consume; fall back to F1 there.
        use_similarity = "repoqa" in dataset_name
        try:
            scores = evaluate_answer(
                predictions_flat,
                references_flat,
                dataset_name,
                "qa",
                similarity=use_similarity,
            )
        except Exception as exc:
            print(f"  evaluate_answer failed ({exc}), falling back to F1")
            scores = [
                f1_score(pred, ref)
                for pred, ref in zip(predictions_flat, references_flat)
            ]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        total_evaluated = sum(len(entry["predictions"]) for entry in per_sample)
        if scores and len(scores) == total_evaluated:
            cursor = 0
            for entry in per_sample:
                num_turns = len(entry["predictions"])
                turn_scores = scores[cursor:cursor + num_turns]
                entry["turn_scores"] = [float(s) for s in turn_scores]
                # 0.5 covers both binary (0/1) and soft (rouge/f1) metrics.
                entry["turn_correct"] = [bool(s >= 0.5) for s in turn_scores]
                cursor += num_turns

        print(
            f"\n  [{dataset_name}]  {len(scores)} QA pairs  →  "
            f"avg score: {avg_score * 100:.2f}%"
        )
        for entry in per_sample[:3]:
            print(f"    Sample {entry['sample_idx']}:")
            for turn_idx in range(min(2, len(entry["predictions"]))):
                tag = ""
                if "turn_scores" in entry and turn_idx < len(entry["turn_scores"]):
                    verdict = (
                        "correct" if entry["turn_correct"][turn_idx] else "wrong"
                    )
                    tag = (
                        f"  [score={entry['turn_scores'][turn_idx]:.2f} "
                        f"{verdict}]"
                    )
                pred_preview = entry["predictions"][turn_idx][:100]
                gt_preview = entry["ground_truths"][turn_idx][:100]
                print(f"      T{turn_idx} pred{tag}: {pred_preview}")
                print(f"      T{turn_idx} gt   : {gt_preview}")

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
        "dataset": dataset_name,
        "model": args.model_path,
        "compression_algo": args.compression_scorer,
        "compression_level": args.compression_level,
        "ratio": args.ratio,
        "page_group_size": args.page_group_size,
        "max_tokens": max_tokens,
        "num_samples": len(per_sample),
        "total_qa_pairs": len(scores),
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
        description="Offline SCBench benchmark for Tangram on vLLM",
    )

    # Model / engine (shared with benchmark_ruler.py).
    add_engine_args(parser)

    # Dataset.
    parser.add_argument(
        "-d", "--data",
        type=str, required=True,
        help=(
            "Dataset name or group (e.g. scbench_kv, scbench_many_shot, squad, "
            "gsm, short, mid, long, all). See scbench_local.get_data_list."
        ),
    )
    parser.add_argument(
        "--num", type=int, default=100,
        help="Number of samples to evaluate per dataset.",
    )
    parser.add_argument(
        "--idx", type=int, default=0,
        help="Start index into the dataset.",
    )
    parser.add_argument(
        "--max-questions", type=int, default=None,
        help="Limit questions per context (default: use all).",
    )
    parser.add_argument(
        "--single-turn", action="store_true", default=False,
        help="Force single-turn: context + first question only.",
    )
    parser.add_argument(
        "--force-exact-tokens", action="store_true", default=False,
        help="Force exactly max_tokens per turn (min_tokens=max, ignore_eos).",
    )

    # Compression (FastKVZip prefill-with-eviction; shared with benchmark_ruler.py).
    add_compression_args(parser)

    # Output.
    parser.add_argument(
        "--output-dir", type=str, default="./results_scbench",
        help="Directory to save evaluation results.",
    )
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Resume mode: skip any dataset whose result JSON already exists "
             "for this (ratio, page_group). If every dataset for the ratio is "
             "already done, the model is not even loaded. Lets an interrupted "
             "sweep be re-run with the same command to fill only the missing "
             "cells.",
    )

    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True,
    )
    model_basename = os.path.basename(args.model_path.rstrip("/"))
    dataset_names = get_data_list(args.data, model_basename, args.max_model_len)

    tag_suffix = f"_{args.tag}" if args.tag else ""

    def save_path_for(dataset_name: str) -> str:
        return os.path.join(
            args.output_dir, dataset_name,
            f"{model_basename}_r{args.ratio}_pg{args.page_group_size}"
            f"{tag_suffix}.json",
        )

    # Resume mode: drop datasets already computed for this (ratio, page_group)
    # so an interrupted sweep continues with the same command. When nothing is
    # pending, skip the model load entirely (it allocates the GPU).
    if args.skip_existing:
        done = [d for d in dataset_names if os.path.exists(save_path_for(d))]
        if done:
            print(f"[resume] ratio={args.ratio}: skipping {len(done)} existing "
                  f"({', '.join(done)})")
        dataset_names = [d for d in dataset_names
                         if not os.path.exists(save_path_for(d))]
        if not dataset_names:
            print(f"[resume] ratio={args.ratio}: all datasets done; "
                  "skipping model load.")
            return

    llm = build_llm(args)

    for dataset_name in dataset_names:
        result = run_dataset(
            llm=llm,
            tokenizer=tokenizer,
            model_basename=model_basename,
            dataset_name=dataset_name,
            args=args,
        )
        if result is None:
            continue

        save_path = save_path_for(dataset_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Results saved → {save_path}")
        print("-" * 70)

    print("\n" + "=" * 70)
    print("  All evaluations finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()
