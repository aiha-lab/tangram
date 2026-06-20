"""Isolate WHERE long-context retrieval fails for gemma-3: head-group paging,
compression, or neither.

Builds a ~9K-token prompt (just over one 8192 compression chunk, so the
multi-chunk + multi-block paths are exercised) of repetitive filler with ONE
distinctive easily-retrievable fact embedded in the middle. A vanilla gemma-3
must retrieve it; if a tangram config fails the SAME easy retrieval, that config
is where the bug is.

Run matrix (compare ``retrieved`` and token_ids):
  pg=0, ratio=1.0   vanilla (head-group OFF, no compression)  -- control
  pg>0, ratio=1.0   head-group paging, no compression          -- paging bug?
  pg>0, ratio<1.0   head-group + FastKVZip compression         -- compression bug?
"""

import argparse
import json

from vllm import LLM, SamplingParams

SECRET = "ZEBRA-9173-QUOKKA"
FILLER = ("The weather report noted mild temperatures and scattered clouds "
          "across the northern plains throughout the long quiet afternoon. ")


def build_prompt(repeats: int) -> str:
    head = FILLER * (repeats // 2)
    tail = FILLER * (repeats - repeats // 2)
    fact = f"\n\nIMPORTANT FACT: The secret access code is {SECRET}.\n\n"
    q = ("\n\nBased on the text above, what is the secret access code? "
         "Answer with only the code.")
    return head + fact + tail + q


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/raid/LLM/gemma-3-12b-it")
    p.add_argument("--page-group-size", type=int, required=True)  # 0 = vanilla
    p.add_argument("--ratio", type=float, default=1.0)
    p.add_argument("--out", required=True)
    p.add_argument("--repeats", type=int, default=620)  # ~9K tokens
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    p.add_argument("--num-copies", type=int, default=1,
                   help="Submit N identical prompts concurrently (concurrency "
                        "test): retrieval rate at batch N vs N=1 isolates "
                        "concurrency-dependent corruption.")
    args = p.parse_args()

    pgs = args.page_group_size if args.page_group_size > 0 else None
    prompt = build_prompt(args.repeats)
    text = ("<bos><start_of_turn>user\nYou are a helpful assistant.\n\n"
            + prompt + "<end_of_turn>\n<start_of_turn>model\n")

    kw = dict(
        model=args.model, page_group_size=pgs, dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True,
    )
    if pgs is not None and args.ratio < 1.0:
        kw.update(compression_ratio=args.ratio,
                  compression_chunk_size=8192, compression_window_size=4096,
                  compression_n_sink_tokens=32, compression_floor_min=0)
    llm = LLM(**kw)
    outs = llm.generate([text] * args.num_copies,
                        SamplingParams(temperature=0.0, max_tokens=32))
    retrieved = [SECRET in o.outputs[0].text for o in outs]
    n_ok = sum(retrieved)
    res = dict(page_group_size=pgs, ratio=args.ratio, num_copies=args.num_copies,
               prompt_tokens=len(outs[0].prompt_token_ids),
               retrieved_count=n_ok, retrieval_rate=n_ok / len(outs),
               sample_texts=[o.outputs[0].text[:40] for o in outs[:6]])
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[longctx] pg={pgs} ratio={args.ratio} N={args.num_copies} "
          f"retrieved={n_ok}/{len(outs)} ({100*n_ok/len(outs):.0f}%) "
          f"samples={res['sample_texts'][:3]}")


if __name__ == "__main__":
    main()
