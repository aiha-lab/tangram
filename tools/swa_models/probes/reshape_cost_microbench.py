"""Quantify the M18.0 decode fast-path cost: value reshape-copy vs the
forward_split alternative, on gemma-3-12b shapes.

The decode fast-path runs once per layer per decode step with
``num_tokens == decode_batch`` (one query token per running request), so the
relevant axis is batch size, not context length.

Compares, per layer, at several decode batch sizes:
  (1) reshape of a NON-contiguous value  (option A — current fix, the copy)
  (2) view   of a CONTIGUOUS value        (what a contiguous input costs: ~0)
  (3) fused QKV GEMM + split              (option A's projection: 1 matmul)
  (4) three separate QKV GEMMs            (option B forward_split: 3 matmuls)
"""

import torch

HIDDEN = 3840          # gemma-3-12b hidden_size
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 256
Q_SIZE = NUM_HEADS * HEAD_DIM       # 4096
KV_SIZE = NUM_KV_HEADS * HEAD_DIM   # 2048
QKV_OUT = Q_SIZE + 2 * KV_SIZE      # 8192
NUM_LAYERS = 48
DTYPE = torch.bfloat16
DEV = "cuda"
ITERS = 200


def bench(fn) -> float:
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS  # ms per call


def main() -> None:
    w_qkv = torch.randn(QKV_OUT, HIDDEN, dtype=DTYPE, device=DEV)
    w_q = w_qkv[:Q_SIZE].contiguous()
    w_k = w_qkv[Q_SIZE:Q_SIZE + KV_SIZE].contiguous()
    w_v = w_qkv[Q_SIZE + KV_SIZE:].contiguous()

    print(f"gemma-3-12b shapes | {NUM_LAYERS} layers | dtype={DTYPE}")
    print(f"{'batch':>6} | {'reshape(noncontig V)':>20} | {'view(contig V)':>15} | "
          f"{'fused GEMM+split':>16} | {'3 split GEMMs':>13}   (per-LAYER us; "
          f"x{NUM_LAYERS} layers in [])")
    for batch in (1, 8, 16, 64, 256):
        x = torch.randn(batch, HIDDEN, dtype=DTYPE, device=DEV)

        # Non-contiguous value: exactly how qkv.split hands it out.
        qkv = x @ w_qkv.t()
        _, _, v_noncontig = qkv.split([Q_SIZE, KV_SIZE, KV_SIZE], dim=-1)
        v_contig = v_noncontig.contiguous()

        t_reshape = bench(
            lambda: v_noncontig.reshape(batch * NUM_KV_HEADS, 1, HEAD_DIM))
        t_view = bench(
            lambda: v_contig.reshape(batch * NUM_KV_HEADS, 1, HEAD_DIM))

        def fused():
            o = x @ w_qkv.t()
            return o.split([Q_SIZE, KV_SIZE, KV_SIZE], dim=-1)

        def split3():
            return (x @ w_q.t(), x @ w_k.t(), x @ w_v.t())

        t_fused = bench(fused)
        t_split3 = bench(split3)

        us = 1000.0
        print(f"{batch:>6} | {t_reshape*us:>9.2f} [{t_reshape*us*NUM_LAYERS:>7.1f}] | "
              f"{t_view*us:>15.2f} | {t_fused*us:>16.2f} | {t_split3*us:>13.2f}")


if __name__ == "__main__":
    main()
