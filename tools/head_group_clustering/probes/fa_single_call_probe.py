"""M17.4 design probe: read a whole layer's KV heads -- each sitting at a
DIFFERENT cluster column -- in ONE FlashAttention call (no per-head overhead).

Motivation. The naive cross-layer read issues one FA call per (layer, kv_head)
because a single call fixes one within-block column (the base pointer). That is
H calls/layer -- unacceptable. The structural fix is to TRANSPOSE the page so the
column dimension sits OUTSIDE block_size:

    column-major page:  [num_phys_blocks, page_group_size, block_size, head_size]

Now each (phys_block, column) pair is a fully contiguous, standard single-head
block, addressable by a flat "virtual block id":

    virtual_block_id = phys_block * page_group_size + column

A cluster owns a set of PHYSICAL blocks (budget shared across its members). Member
at column c reads/writes virtual blocks {phys*g + c}. Because the column is folded
into the virtual block id -- which is per-sequence, per-entry in the block table --
ONE FA call can serve every kv_head of a layer, each sequence pointing at its own
column via its virtual block ids. The cache view passed to FA is the standard
[num_phys_blocks * g, block_size, 1, head_size] single-KV-head paged layout, so no
kernel change and full coalescing.

This probe builds a column-major cache, places three "heads" at DIFFERENT columns
(two of them SHARING physical blocks, i.e. one cluster), and reads all three in a
SINGLE flash_attn_varlen_func call via a virtual block table. It compares each
sequence against a per-head manual reference.

Verdict: if the single mixed-column call matches the reference, cross-layer
clustering reads at 1 call/layer with the stock kernel.

    CUDA_VISIBLE_DEVICES=1 python -m tools.head_group_clustering.probes.fa_single_call_probe
"""
from __future__ import annotations

import torch

from vllm.vllm_flash_attn import flash_attn_varlen_func


def reference_attention(q, k_gathered, v_gathered, scale):
    """Manual full attention; q [q_len, nq, hs], k/v [k_len, hs]."""
    out = torch.empty_like(q)
    for h in range(q.shape[1]):
        scores = (q[:, h, :].float() @ k_gathered.float().T) * scale
        out[:, h, :] = (torch.softmax(scores, dim=-1) @ v_gathered.float()).to(q.dtype)
    return out


def run_probe(fa_version: int) -> dict:
    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    g = 4                # page_group_size (cluster width)
    head_size = 64
    block_size = 16
    num_phys_blocks = 8
    scale = head_size ** -0.5

    # Column-major page: column dimension OUTSIDE block_size.
    cache_cm = torch.randn(num_phys_blocks, g, block_size, head_size,
                           device=device, dtype=dtype)
    value_cm = torch.randn_like(cache_cm)

    # Standard single-KV-head paged view over virtual blocks (phys*g + col).
    key_view = cache_cm.reshape(num_phys_blocks * g, block_size, 1, head_size)
    value_view = value_cm.reshape(num_phys_blocks * g, block_size, 1, head_size)
    assert key_view.is_contiguous(), "virtual-block view must be contiguous"

    # Three heads of one layer. Heads 0 and 1 share physical blocks [0,1,2]
    # (they are one cluster, columns 0 and 2); head 2 lives in blocks [3,4],
    # column 3. This exercises both distinct columns and shared physical blocks.
    heads = [
        {"phys_blocks": [0, 1, 2], "col": 0, "k_len": 40},
        {"phys_blocks": [0, 1, 2], "col": 2, "k_len": 36},
        {"phys_blocks": [3, 4],    "col": 3, "k_len": 24},
    ]
    q_len, nq_per_head = 4, 1  # MQA per kv head
    max_blocks = max(len(h["phys_blocks"]) for h in heads)

    # Virtual block table: row per head, entry = phys*g + col.
    block_table = torch.zeros(len(heads), max_blocks, dtype=torch.int32, device=device)
    for i, h in enumerate(heads):
        for j, pb in enumerate(h["phys_blocks"]):
            block_table[i, j] = pb * g + h["col"]

    q = torch.randn(q_len * len(heads), nq_per_head, head_size,
                    device=device, dtype=dtype)
    cu_seqlens_q = torch.arange(0, q_len * (len(heads) + 1), q_len,
                                dtype=torch.int32, device=device)
    seqused_k = torch.tensor([h["k_len"] for h in heads],
                             dtype=torch.int32, device=device)

    out = torch.empty_like(q)
    flash_attn_varlen_func(
        q=q, k=key_view, v=value_view, out=out,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=q_len,
        seqused_k=seqused_k, max_seqlen_k=int(seqused_k.max()),
        softmax_scale=scale, causal=False,
        block_table=block_table, fa_version=fa_version)

    # Per-head reference from the column-major cache.
    worst = 0.0
    for i, h in enumerate(heads):
        k_g = torch.empty(h["k_len"], head_size, device=device, dtype=dtype)
        v_g = torch.empty(h["k_len"], head_size, device=device, dtype=dtype)
        for pos in range(h["k_len"]):
            pb = h["phys_blocks"][pos // block_size]
            off = pos % block_size
            k_g[pos] = cache_cm[pb, h["col"], off, :]
            v_g[pos] = value_cm[pb, h["col"], off, :]
        ref = reference_attention(q[i * q_len:(i + 1) * q_len], k_g, v_g, scale)
        diff = (out[i * q_len:(i + 1) * q_len].float() - ref.float()).abs().max().item()
        worst = max(worst, diff)
    return {"fa_version": fa_version, "num_heads_in_one_call": len(heads),
            "worst_vs_ref": worst}


def run_write_probe() -> dict:
    """Write a whole layer's heads -- different columns, some sharing physical
    blocks -- in ONE reshape_and_cache call, by encoding the column into the
    virtual block id of each (token, head) slot.

    slot = virtual_block * block_size + offset = (phys_block * g + col) * block_size + offset

    The cache view is the standard [num_phys*g, block_size, 1, head_size] virtual-
    block layout; each slot entry is one (token, head). We verify every head lands
    at its column and untouched columns stay intact.
    """
    from vllm.attention.utils.fa_utils import reshape_and_cache_flash

    torch.manual_seed(2)
    device, dtype = "cuda", torch.bfloat16
    g, head_size, block_size, num_phys_blocks = 4, 64, 16, 8
    cache_cm = torch.randn(num_phys_blocks, g, block_size, head_size,
                           device=device, dtype=dtype)
    value_cm = torch.randn_like(cache_cm)
    before = cache_cm.clone()

    key_view = cache_cm.reshape(num_phys_blocks * g, block_size, 1, head_size)
    value_view = value_cm.reshape(num_phys_blocks * g, block_size, 1, head_size)

    # Two heads share physical block 0 (one cluster) at columns 0 and 2; a third
    # head writes block 1 column 3. Each writes `n` tokens into its column.
    writes = [{"phys_block": 0, "col": 0, "n": 10},
              {"phys_block": 0, "col": 2, "n": 10},
              {"phys_block": 1, "col": 3, "n": 8}]
    keys, slots = [], []
    for w in writes:
        keys.append(torch.randn(w["n"], 1, head_size, device=device, dtype=dtype))
        vb = w["phys_block"] * g + w["col"]
        slots.append(torch.arange(w["n"], dtype=torch.int64, device=device)
                     + vb * block_size)
    key_new = torch.cat(keys, dim=0)
    value_new = torch.randn_like(key_new)
    slot_mapping = torch.cat(slots, dim=0)
    scale = torch.ones(1, device=device, dtype=torch.float32)

    reshape_and_cache_flash(key_new, value_new, key_view, value_view,
                            slot_mapping, "auto", scale, scale)

    # Each head's written column must equal its input.
    target = 0.0
    start = 0
    touched = set()
    for w, kk in zip(writes, keys):
        got = cache_cm[w["phys_block"], w["col"], :w["n"], :]
        target = max(target, (got.float() - kk[:, 0, :].float()).abs().max().item())
        touched.add((w["phys_block"], w["col"]))
        start += w["n"]
    # Untouched (block, column) cells must be unchanged.
    other = 0.0
    for b in range(num_phys_blocks):
        for c in range(g):
            if (b, c) in touched:
                continue
            other = max(other, (cache_cm[b, c].float()
                                - before[b, c].float()).abs().max().item())
    return {"target_diff": target, "other_diff": other}


def main() -> int:
    if not torch.cuda.is_available():
        print("[probe] CUDA not available.")
        return 1
    print(f"[probe] device={torch.cuda.get_device_name(0)}")
    major = torch.cuda.get_device_capability(0)[0]
    versions = (2, 3) if major >= 9 else (2,)
    if major < 9:
        print("[probe] device < sm90: FA2 only.")
    w = run_write_probe()
    w_ok = w["target_diff"] == 0 and w["other_diff"] == 0
    print(f"[probe] WRITE: 3 heads (mixed columns, shared phys block) in ONE call "
          f"-> target_diff={w['target_diff']:.2e} other_diff={w['other_diff']:.2e} "
          f"{'OK' if w_ok else 'FAIL'}")

    fail = not w_ok
    for v in versions:
        try:
            r = run_probe(v)
        except Exception as exc:  # noqa: BLE001
            print(f"[probe] fa_version={v}: ERROR {exc!r}")
            fail = True
            continue
        ok = r["worst_vs_ref"] <= 5e-2
        print(f"[probe] fa_version={v}: "
              f"{r['num_heads_in_one_call']} heads (mixed columns) in ONE call -> "
              f"worst_vs_ref={r['worst_vs_ref']:.2e} "
              f"{'OK (1x calls feasible)' if ok else 'FAIL'}")
        fail = fail or not ok
    return 2 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
