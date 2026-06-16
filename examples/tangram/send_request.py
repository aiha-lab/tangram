# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Send SQuAD requests to a running vLLM OpenAI-compatible server.

Pairs with ``online_serving.sh`` (which starts a SnapKV-compression server).
The SQuAD data, prompt template, and query wrapping are reused verbatim from
``benchmarks/tangram`` (``scbench_local``), so a request here is byte-identical
to one turn of ``benchmark_scbench.py`` — only the transport differs (HTTP
instead of the in-process ``LLM``).

Because ``scbench_local.template`` already emits the model's chat special
tokens (e.g. ``<|im_start|>`` for Qwen), the prompt must go to the raw
``/v1/completions`` endpoint, NOT ``/v1/chat/completions`` (which would wrap
it a second time).

Example:
    # 1) start the server
    bash online_serving.sh
    # 2) in another shell
    python send_request.py --num 10
"""

import argparse
import asyncio
import os
import sys

# Reuse the engine's own SCBench helpers (benchmarks/tangram/scbench_local.py):
# the SQuAD loader, per-model prompt template, and generation-length preset.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "benchmarks", "tangram"))
from scbench_local import (  # noqa: E402
    load_dataset_all,
    set_gen_length,
    template,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1", help="server host")
    p.add_argument("--port", type=int, default=8000, help="server port")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                   help="served model name (must match what the server reports "
                        "at /v1/models; also selects the prompt template).")
    p.add_argument("--num", type=int, default=10,
                   help="number of SQuAD contexts to load (n_data).")
    p.add_argument("--max-questions", type=int, default=1,
                   help="questions per context to send (each is an independent "
                        "request carrying the full context).")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="output tokens per request; default = the dataset preset "
                        "(squad -> 256, see scbench_local.set_gen_length).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="sampling temperature (0 = greedy).")
    p.add_argument("--request-rate", type=float, default=float("inf"),
                   help="requests dispatched per second (default inf = fire all "
                        "at once).")
    p.add_argument("--max-concurrency", type=int, default=None,
                   help="max in-flight requests at a time (default: unlimited).")
    p.add_argument("--force-exact-tokens", action="store_true",
                   help="emit exactly max_tokens per request (sets min_tokens="
                        "max_tokens, suppressing early EOS), matching "
                        "benchmark_scbench's --force-exact-tokens.")
    p.add_argument("--api-key", default="EMPTY",
                   help="API key sent to the server (vLLM ignores it by default).")
    return p.parse_args()


def build_prompts(model: str, dataset: list[dict],
                  max_questions: int) -> list[dict]:
    """One prompt per (context, question), matching benchmark_scbench's turn
    prompt: prefix + context + "\\n\\nQ: {question}" + postfix."""
    prefix, postfix = template(model, "squad")
    prompts: list[dict] = []
    for sample in dataset:
        context = sample["context"]
        for question in list(sample["question"])[:max_questions]:
            if not question:
                continue
            prompt = f"{prefix}{context}\n\nQ: {question.strip()}{postfix}"
            prompts.append({"question": question, "prompt": prompt})
    return prompts


def main() -> int:
    args = parse_args()

    # SQuAD ignores the tokenizer argument in scbench_local; pass None.
    dataset = load_dataset_all("squad", tokenizer=None, n_data=args.num)
    prompts = build_prompts(args.model, dataset, args.max_questions)
    max_tokens = args.max_tokens or set_gen_length("squad")

    base_url = f"http://{args.host}:{args.port}/v1"
    print(f"\n[send] {len(prompts)} request(s) -> {base_url} "
          f"(model={args.model}, max_tokens={max_tokens}, "
          f"request_rate={args.request_rate}, "
          f"max_concurrency={args.max_concurrency})\n")

    results = asyncio.run(_send_all(args, prompts, base_url, max_tokens))

    for i, (item, text) in enumerate(zip(prompts, results)):
        print(f"--- [{i}] Q: {item['question']}")
        print(f"    A: {text}")

    return 0


async def _send_all(args: argparse.Namespace, prompts: list[dict],
                    base_url: str, max_tokens: int) -> list[str]:
    """Dispatch requests at ``--request-rate`` per second (inf = all at once)
    and cap in-flight count at ``--max-concurrency`` (None = unlimited). The
    server's scheduler batches whatever overlaps. Returns predictions in
    prompt order."""
    from openai import AsyncOpenAI

    # min_tokens is a vLLM extension (not standard OpenAI), so it rides in
    # extra_body. min_tokens == max_tokens forces exactly max_tokens out.
    extra_body = (
        {"min_tokens": max_tokens} if args.force_exact_tokens else None)

    client = AsyncOpenAI(base_url=base_url, api_key=args.api_key,
                         max_retries=0)
    # None concurrency -> a semaphore big enough to never block.
    sem = asyncio.Semaphore(args.max_concurrency or len(prompts) or 1)

    async def one(item: dict) -> str:
        async with sem:
            resp = await client.completions.create(
                model=args.model,
                prompt=item["prompt"],
                max_tokens=max_tokens,
                temperature=args.temperature,
                extra_body=extra_body,
            )
        return resp.choices[0].text.strip()

    try:
        tasks = []
        for item in prompts:
            # Space out dispatch for a finite rate; inf -> no delay.
            if args.request_rate != float("inf"):
                await asyncio.sleep(1.0 / args.request_rate)
            tasks.append(asyncio.create_task(one(item)))
        # gather preserves input order regardless of completion order.
        return await asyncio.gather(*tasks)
    finally:
        await client.close()


if __name__ == "__main__":
    raise SystemExit(main())
