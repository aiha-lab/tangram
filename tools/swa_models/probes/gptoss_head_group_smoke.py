"""M18.4 smoke probe — gpt-oss-20b under head-group paging (compression off).

Mirrors ``gemma3_head_group_smoke.py`` for the second SWA target. gpt-oss adds
two things gemma-3 lacks: a Mixture-of-Experts FFN (MXFP4-quantised) and a
per-head learnable *attention sink*. The sink has no native FlashAttention
support on pre-Hopper GPUs (FA v2 on A100), so the head-group forward must
recover it by the same log-sum-exp rescaling the FastKVzip baseline uses
(``o *= sigmoid(lse - sink)``).

Usage (one model per process):
  --page-group-size 0   base vLLM reference (sink handled by stock backend)
  --page-group-size 2   head-group paging active (sink handled by our forward)

The companion shell script runs both and diffs token ids. Exit gate for M18.4
correctness: the head-group run must be token-identical to base, proving the
sink correction in the head-group path is numerically exact.

This probe also reports the attention backend vLLM selected, because on A100
the stock FlashAttention backend refuses to construct with sinks (it asserts
``flash_attn_supports_sinks()``), so base vLLM is expected to fall back to a
sink-capable backend (Triton). That fact drives the head-group design.
"""

import argparse
import json

from vllm import LLM, SamplingParams

# Mixed prompts: short factual + longer reasoning so both the sliding-window
# (128) and full-attention (global) layers, and the MoE router, are exercised.
PROMPTS = [
    "The capital of France is",
    "List three primes:",
    "In one sentence, explain why the sky appears blue during the day.",
    "Q: What is 17 multiplied by 23? A:",
]


def _report_attention_backend(llm: LLM) -> str:
    """Return the attention impl class name of the first decoder layer.

    Reaches through the v1 engine into the worker's model to read the actual
    backend that was instantiated, rather than inferring it from logs. Best
    effort: returns "unknown" if the internal layout differs.
    """
    try:
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        for module in model.modules():
            impl = getattr(module, "impl", None)
            if impl is not None and module.__class__.__name__ == "Attention":
                return impl.__class__.__name__
    except Exception as exc:  # noqa: BLE001 — probe diagnostics only
        return f"unknown ({exc})"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/raid/LLM/gpt-oss-20b")
    parser.add_argument(
        "--page-group-size",
        type=int,
        required=True,
        help="Head-group size. 0 means disabled (base vLLM reference).",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    parser.add_argument(
        "--enable-compression",
        action="store_true",
        help="Activate FastKVZip compression (requires page-group-size).",
    )
    parser.add_argument("--compression-ratio", type=float, default=1.0)
    parser.add_argument(
        "--head-group-cluster-map",
        default=None,
        help="Path to a head-group cluster-map .npz. At ratio 1.0 it only "
        "changes physical KV placement, so output must stay token-identical "
        "to a no-map run (the cluster-map correctness gate).",
    )
    parser.add_argument(
        "--compression-gate-path",
        default="/workspace/tangram_impl/FastKVzip-gpt-oss/result_gate/"
        "gpt-oss-20b/q8_dim16_sink16.pt",
    )
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

    if args.head_group_cluster_map:
        extra["head_group_cluster_map"] = args.head_group_cluster_map

    llm = LLM(
        model=args.model,
        page_group_size=page_group_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,  # remove cudagraph variability from the comparison
        **extra,
    )

    backend = _report_attention_backend(llm)
    print(f"[smoke] attention impl = {backend}")

    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outputs = llm.generate(PROMPTS, sampling)

    result = {
        "page_group_size": page_group_size,
        "model": args.model,
        "attention_impl": backend,
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
