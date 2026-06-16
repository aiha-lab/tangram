# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Throughput/latency benchmark for a running vLLM compression server.

Same model as vLLM's own ``vllm bench serve`` (vllm/benchmarks/serve.py): it
does NOT launch a server — it loads a dataset (``--dataset``: squad by default,
or any scbench_* / gsm, via ``benchmarks/tangram/scbench_local.load_dataset_all``,
same loader as ``benchmark_scbench.py``), fires the requests, and measures latency
from the streamed response. The point is to compare COMPRESSION CONFIGS: the
benchmark sees only whatever config the server was started with, so launch a
server (``online_serving.sh`` with the flags you want), run this with a
``--label`` describing that config, then relaunch with the next config. Each
run appends one JSON line to ``--output`` so the configs line up for comparison.

Streamed metrics (per request, then aggregated):
  * TTFT  — time to first token (prefill-bound; where compression shows up most)
  * TPOT  — mean time per output token after the first (decode-bound)
  * E2EL  — end-to-end latency
  * throughput — requests/s and tokens/s over the wall-clock window

For a fair config-to-config comparison, fix the output length with
``--force-exact-tokens`` (every request emits exactly ``--max-tokens``).

Example:
    bash online_serving.sh        # config A
    python benchmark.py --num 200 --force-exact-tokens --label snapkv_r0.3
    # ... relaunch server with config B ...
    python benchmark.py --num 200 --force-exact-tokens --label snapkv_r0.5
    # compare the rows in benchmark_results.jsonl
"""

import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np

# Reuse send_request's prompt construction (identical to benchmarks/tangram)
# and scbench_local's dataset loader / generation-length preset.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from send_request import build_prompts  # noqa: E402
sys.path.insert(0, os.path.join(
    os.path.abspath(os.path.join(_HERE, "..", "..")), "benchmarks", "tangram"))
from scbench_local import load_dataset_all, set_gen_length  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                   help="served model name (must match /v1/models).")
    p.add_argument("--dataset", default="squad",
                   help="dataset to load (passed to scbench_local.load_dataset_all): "
                        "squad, gsm, or any scbench_* (e.g. scbench_kv, scbench_summary, "
                        "scbench_vt, scbench_prefix_suffix, scbench_many_shot).")
    p.add_argument("--num", type=int, default=200,
                   help="number of contexts to load from --dataset.")
    p.add_argument("--max-questions", type=int, default=1,
                   help="questions per context (each is one request).")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="output tokens per request; 0 = --dataset's set_gen_length preset.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--force-exact-tokens", default=True, action="store_true",
                   help="emit exactly max_tokens per request (min_tokens="
                        "max_tokens). Use it so output length is constant across "
                        "configs — otherwise throughput differences are confounded "
                        "by differing answer lengths.")
    p.add_argument("--request-rate", type=float, default=float("inf"),
                   help="requests dispatched per second (default inf = all at once).")
    p.add_argument("--max-concurrency", type=int, default=None,
                   help="max in-flight requests (default: unlimited).")
    p.add_argument("--label", default="",
                   help="free-text tag for this run's server/compression config, "
                        "recorded in the output row (e.g. 'snapkv_r0.3').")
    p.add_argument("--output", default=os.path.join(_HERE, "benchmark_results.jsonl"),
                   help="JSONL file; one result row is appended per run.")
    p.add_argument("--api-key", default="EMPTY")
    return p.parse_args()


async def _bench_one(client, args, prompt: str, max_tokens: int,
                     extra_body, sem) -> dict | None:
    """Stream one completion; return its per-request timings, or None on error."""
    async with sem:
        t0 = time.perf_counter()
        ttft = None
        prompt_tokens = output_tokens = 0
        try:
            stream = await client.completions.create(
                model=args.model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=args.temperature,
                stream=True,
                stream_options={"include_usage": True},
                extra_body=extra_body,
            )
            async for chunk in stream:
                now = time.perf_counter()
                if chunk.choices and chunk.choices[0].text and ttft is None:
                    ttft = now - t0
                if chunk.usage is not None:  # final chunk (include_usage)
                    prompt_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
        except Exception as exc:  # noqa: BLE001 — record failure, keep going
            print(f"  request failed: {exc}")
            return None
    e2e = time.perf_counter() - t0
    if ttft is None:  # no token streamed (e.g. empty output)
        ttft = e2e
    tpot = (e2e - ttft) / (output_tokens - 1) if output_tokens > 1 else 0.0
    return {"ttft": ttft, "e2e": e2e, "tpot": tpot,
            "prompt_tokens": prompt_tokens, "output_tokens": output_tokens}


async def _run(args, prompts, base_url, max_tokens) -> dict:
    from openai import AsyncOpenAI

    extra_body = (
        {"min_tokens": max_tokens} if args.force_exact_tokens else None)
    client = AsyncOpenAI(base_url=base_url, api_key=args.api_key, max_retries=0)
    sem = asyncio.Semaphore(args.max_concurrency or len(prompts) or 1)

    wall_start = time.perf_counter()
    try:
        tasks = []
        for item in prompts:
            if args.request_rate != float("inf"):
                await asyncio.sleep(1.0 / args.request_rate)
            tasks.append(asyncio.create_task(
                _bench_one(client, args, item["prompt"], max_tokens,
                           extra_body, sem)))
        per_req = await asyncio.gather(*tasks)
    finally:
        await client.close()
    wall = time.perf_counter() - wall_start

    ok = [r for r in per_req if r is not None]
    return {"wall": wall, "ok": ok, "n_total": len(prompts)}


def _pcts(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float) * 1000.0  # -> ms
    if a.size == 0:
        return {"mean": 0.0, "median": 0.0, "p99": 0.0}
    return {"mean": float(a.mean()),
            "median": float(np.median(a)),
            "p99": float(np.percentile(a, 99))}


def main() -> int:
    args = parse_args()

    # scbench_* datasets don't use the tokenizer (only gsm does); squad/gsm and
    # all scbench_* names are accepted by load_dataset_all, which raises on others.
    dataset = load_dataset_all(args.dataset, tokenizer=None, n_data=args.num)
    # load_dataset_all honors n_data for squad/gsm, but scbench_* loads its full
    # split; slice uniformly so --num is authoritative across all datasets.
    dataset = dataset[: args.num]
    prompts = build_prompts(args.model, dataset, args.max_questions)
    max_tokens = args.max_tokens or set_gen_length(args.dataset)

    base_url = f"http://{args.host}:{args.port}/v1"
    print(f"\n[bench] {len(prompts)} requests -> {base_url} "
          f"(label={args.label or '-'}, max_tokens={max_tokens}, "
          f"force_exact={args.force_exact_tokens}, "
          f"request_rate={args.request_rate}, "
          f"max_concurrency={args.max_concurrency})\n")

    res = asyncio.run(_run(args, prompts, base_url, max_tokens))
    ok, wall = res["ok"], res["wall"]
    n = len(ok)
    if n == 0:
        print("[bench] all requests failed — is the server up?")
        return 1

    in_tok = sum(r["prompt_tokens"] for r in ok)
    out_tok = sum(r["output_tokens"] for r in ok)
    ttft = _pcts([r["ttft"] for r in ok])
    tpot = _pcts([r["tpot"] for r in ok])
    e2e = _pcts([r["e2e"] for r in ok])

    row = {
        "label": args.label,
        "model": args.model,
        "dataset": args.dataset,
        "num_requests": n,
        "num_failed": res["n_total"] - n,
        "max_tokens": max_tokens,
        "force_exact_tokens": args.force_exact_tokens,
        # inf -> "inf" string so the row stays valid (standard) JSON.
        "request_rate": ("inf" if args.request_rate == float("inf")
                         else args.request_rate),
        "max_concurrency": args.max_concurrency,
        "duration_s": round(wall, 3),
        "total_input_tokens": in_tok,
        "total_output_tokens": out_tok,
        "request_throughput": round(n / wall, 3),
        "output_token_throughput": round(out_tok / wall, 2),
        "total_token_throughput": round((in_tok + out_tok) / wall, 2),
        "ttft_ms": {k: round(v, 2) for k, v in ttft.items()},
        "tpot_ms": {k: round(v, 2) for k, v in tpot.items()},
        "e2e_ms": {k: round(v, 2) for k, v in e2e.items()},
    }

    print("============ Benchmark Result ============")
    print(f"Label:                         {row['label'] or '-'}")
    print(f"Successful / total:            {n} / {res['n_total']}")
    print(f"Benchmark duration (s):        {row['duration_s']}")
    print(f"Total input tokens:            {in_tok}")
    print(f"Total generated tokens:        {out_tok}")
    print(f"Request throughput (req/s):    {row['request_throughput']}")
    print(f"Output token throughput (t/s): {row['output_token_throughput']}")
    print(f"Total token throughput (t/s):  {row['total_token_throughput']}")
    print("----------------- TTFT (ms) -----------------")
    print(f"  mean {ttft['mean']:.2f}  median {ttft['median']:.2f}  p99 {ttft['p99']:.2f}")
    print("----------------- TPOT (ms) -----------------")
    print(f"  mean {tpot['mean']:.2f}  median {tpot['median']:.2f}  p99 {tpot['p99']:.2f}")
    print("----------------- E2EL (ms) -----------------")
    print(f"  mean {e2e['mean']:.2f}  median {e2e['median']:.2f}  p99 {e2e['p99']:.2f}")
    print("==========================================")

    with open(args.output, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\n[bench] appended result to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
