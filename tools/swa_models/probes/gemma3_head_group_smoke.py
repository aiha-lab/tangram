"""M18.0 smoke probe — gemma-3-12b-it under head-group paging (compression off).

Generates greedy continuations for a fixed prompt set and writes the produced
token ids to a JSON file. Run once with ``--page-group-size 0`` (base vLLM, the
reference) and once with ``--page-group-size 2`` (head-group paging active); the
companion shell script diffs the two outputs.

Exit gate for M18.0: the head-group run must be token-identical to the base run,
proving that the per-layer sliding-window forward path (full KV retained, window
applied inside FlashAttention) is numerically equivalent to stock attention for a
hybrid sliding/full-attention model.
"""

import argparse
import json

from vllm import LLM, SamplingParams

# Mixed prompts: short factual + longer context so both sliding-window (1024)
# and full-attention (global) layers are exercised. Kept well under the model's
# window so behaviour is unambiguous; correctness, not length, is the target.
PROMPTS = [
    "The capital of France is",
    "List three primes:",
    "In one sentence, explain why the sky appears blue during the day.",
    "Q: What is 17 multiplied by 23? A:",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/raid/LLM/gemma-3-12b-it")
    parser.add_argument(
        "--page-group-size",
        type=int,
        required=True,
        help="Head-group size. 0 means disabled (base vLLM reference).",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--enable-compression",
        action="store_true",
        help="Activate FastKVZip compression (requires page-group-size).",
    )
    parser.add_argument("--compression-ratio", type=float, default=1.0)
    parser.add_argument("--compression-gate-path", default="fastkvzip")
    args = parser.parse_args()

    # ``page_group_size=None`` is the explicit "disabled" sentinel that keeps
    # the base vLLM hot path (zero-overhead). A CLI 0 maps to that sentinel.
    page_group_size = args.page_group_size if args.page_group_size > 0 else None

    extra: dict = {}
    if args.enable_compression:
        extra.update(
            compression_ratio=args.compression_ratio,
            compression_gate_path=args.compression_gate_path,
        )

    llm = LLM(
        model=args.model,
        page_group_size=page_group_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.45,
        enforce_eager=True,  # remove cudagraph variability from the comparison
        **extra,
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outputs = llm.generate(PROMPTS, sampling)

    result = {
        "page_group_size": page_group_size,
        "model": args.model,
        "generations": [
            {
                "prompt": out.prompt,
                "token_ids": list(out.outputs[0].token_ids),
                "text": out.outputs[0].text,
            }
            for out in outputs
        ],
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[smoke] wrote {args.out} (page_group_size={page_group_size})")


if __name__ == "__main__":
    main()
