"""M17.3 risk probe: can FlashAttention read ONE KV head at a column offset
inside a wider (page_group_size) paged cache page?

Cross-layer head clustering stores ``page_group_size`` KV heads -- belonging to
*different* layers -- in the columns of one physical page. When a layer computes
attention it must read only ITS column of the shared page (the other columns
belong to other layers and carry unrelated KV). The vLLM Python wrapper
(`flash_attn_varlen_func`) only forces last-dim contiguity, so a column-offset
view ``key_cache[:, :, col:col+1, :]`` reaches the CUDA kernel *strided* (block
and per-token strides still span the full page; the base pointer is offset by
``col * head_size``). Whether that reads correctly depends on whether the
compiled FA2/FA3 kernel honours the tensor's strides or assumes a packed layout.
The flash-attention CUDA source is not vendored here, so this can only be settled
empirically.

This probe compares, for a single KV head:
  * REF      -- manual softmax attention over KV gathered from column ``col``.
  * PACKED   -- flash_attn_varlen_func on a contiguous single-head cache (the
               known-correct FA usage; isolates kernel-vs-reference numerics).
  * STRIDED  -- flash_attn_varlen_func on the column-offset view of the wide page
               (the cross-layer read we need).

Verdict: if STRIDED matches PACKED (and REF) within tolerance, the cross-layer
per-column read is feasible with the stock kernel. If STRIDED diverges, the
kernel assumes a packed layout and a custom kernel / fallback is required.

Run on an idle GPU, e.g.:
    CUDA_VISIBLE_DEVICES=1 python -m tools.head_group_clustering.probes.fa_column_offset_probe
"""
from __future__ import annotations

import torch

from vllm.vllm_flash_attn import flash_attn_varlen_func


def reference_attention(
    q: torch.Tensor,       # [q_len, num_q_heads, head_size]
    k_gathered: torch.Tensor,  # [k_len, head_size]
    v_gathered: torch.Tensor,  # [k_len, head_size]
    scale: float,
) -> torch.Tensor:
    """Manual full (non-causal) attention; the single KV head is shared by all
    query heads (GQA/MQA). Returns [q_len, num_q_heads, head_size]."""
    q_len, num_q_heads, head_size = q.shape
    out = torch.empty_like(q)
    for h in range(num_q_heads):
        scores = (q[:, h, :].float() @ k_gathered.float().T) * scale  # [q_len, k_len]
        probs = torch.softmax(scores, dim=-1)
        out[:, h, :] = (probs @ v_gathered.float()).to(q.dtype)
    return out


def gather_column(
    cache: torch.Tensor,        # [num_blocks, block_size, page_group_size, head_size]
    block_table_row: torch.Tensor,  # [num_blocks_used] int
    k_len: int,
    col: int,
    block_size: int,
) -> torch.Tensor:
    """Gather the logical sequence (length k_len) for one column into a
    contiguous [k_len, head_size] tensor, following the block table."""
    head_size = cache.shape[-1]
    out = torch.empty((k_len, head_size), dtype=cache.dtype, device=cache.device)
    for pos in range(k_len):
        block = int(block_table_row[pos // block_size])
        offset = pos % block_size
        out[pos] = cache[block, offset, col, :]
    return out


def run_probe(fa_version: int) -> dict:
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    page_group_size = 4      # KV heads per physical page (the cluster width)
    head_size = 64
    block_size = 16
    num_blocks = 8
    num_q_heads = 2          # query heads sharing the one KV head (GQA ratio 2)
    q_len = 4
    k_len = 40               # spans 3 blocks: 16 + 16 + 8
    col = 2                  # the target column (this layer's head in the cluster)
    scale = head_size ** -0.5

    # Full wide page cache: every column carries distinct random KV so a kernel
    # that ignores the column offset would read the wrong data and fail.
    key_cache = torch.randn(
        num_blocks, block_size, page_group_size, head_size,
        device=device, dtype=dtype)
    value_cache = torch.randn_like(key_cache)

    q = torch.randn(q_len, num_q_heads, head_size, device=device, dtype=dtype)

    # One sequence using blocks [0, 1, 2] for its 40 positions.
    block_table = torch.tensor([[0, 1, 2, 0, 0, 0, 0, 0]],
                               dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    seqused_k = torch.tensor([k_len], dtype=torch.int32, device=device)

    # ----- REF: manual attention over the gathered column -----
    k_gathered = gather_column(key_cache, block_table[0], k_len, col, block_size)
    v_gathered = gather_column(value_cache, block_table[0], k_len, col, block_size)
    ref = reference_attention(q, k_gathered, v_gathered, scale)

    fa_kwargs = dict(
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=q_len,
        seqused_k=seqused_k,
        max_seqlen_k=k_len,
        softmax_scale=scale,
        causal=False,
        block_table=block_table,
        fa_version=fa_version,
    )

    # ----- PACKED: contiguous single-head cache (known-correct FA usage) -----
    key_packed = key_cache[:, :, col:col + 1, :].contiguous()
    value_packed = value_cache[:, :, col:col + 1, :].contiguous()
    out_packed = torch.empty(q_len, num_q_heads, head_size,
                             device=device, dtype=dtype)
    flash_attn_varlen_func(q=q, k=key_packed, v=value_packed, out=out_packed,
                           **fa_kwargs)

    # ----- STRIDED: column-offset view of the wide page (the cross-layer read) -----
    key_view = key_cache[:, :, col:col + 1, :]
    value_view = value_cache[:, :, col:col + 1, :]
    assert key_view.stride(-1) == 1, "last dim must stay contiguous"
    assert not key_view.is_contiguous(), "probe must exercise a strided view"
    out_strided = torch.empty(q_len, num_q_heads, head_size,
                              device=device, dtype=dtype)
    flash_attn_varlen_func(q=q, k=key_view, v=value_view, out=out_strided,
                           **fa_kwargs)

    def max_abs_diff(a, b):
        return (a.float() - b.float()).abs().max().item()

    return {
        "fa_version": fa_version,
        "packed_vs_ref": max_abs_diff(out_packed, ref),
        "strided_vs_ref": max_abs_diff(out_strided, ref),
        "strided_vs_packed": max_abs_diff(out_strided, out_packed),
        "key_view_strides": tuple(key_view.stride()),
        "key_view_shape": tuple(key_view.shape),
    }


def run_write_probe() -> dict:
    """Can reshape_and_cache write ONE column of a wide page without touching the
    others, using just a column-offset cache VIEW (no kernel change)?

    A torch tensor's data_ptr already folds in its storage offset, and the triton
    reshape_and_cache kernel reads block/page strides from the cache tensor. So
    passing ``key_cache[:, :, col:col+seg, :]`` as the cache plus a [tokens, seg,
    head_size] key should land exactly in columns [col, col+seg). We verify the
    target column receives the input and every other column is untouched.
    """
    from vllm.attention.utils.fa_utils import reshape_and_cache_flash

    torch.manual_seed(1)
    device = "cuda"
    dtype = torch.bfloat16
    page_group_size, head_size, block_size, num_blocks = 4, 64, 16, 8
    col, num_tokens = 2, 10

    key_cache = torch.randn(num_blocks, block_size, page_group_size, head_size,
                            device=device, dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    before = key_cache.clone()

    key_new = torch.randn(num_tokens, 1, head_size, device=device, dtype=dtype)
    value_new = torch.randn(num_tokens, 1, head_size, device=device, dtype=dtype)
    # Distinct slots within block 0.
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    key_view = key_cache[:, :, col:col + 1, :]
    value_view = value_cache[:, :, col:col + 1, :]
    scale = torch.ones(1, device=device, dtype=torch.float32)
    reshape_and_cache_flash(key_new, value_new, key_view, value_view,
                            slot_mapping, "auto", scale, scale)

    # Target column at the written slots must equal the input.
    written = key_cache[0, :num_tokens, col, :]
    target_diff = (written.float() - key_new[:, 0, :].float()).abs().max().item()
    # Every other column everywhere must be byte-identical to before.
    other_cols = [c for c in range(page_group_size) if c != col]
    other_diff = (key_cache[:, :, other_cols, :].float()
                  - before[:, :, other_cols, :].float()).abs().max().item()
    return {"target_col_diff": target_diff, "other_cols_diff": other_diff}


def main() -> int:
    if not torch.cuda.is_available():
        print("[probe] CUDA not available; cannot run.")
        return 1
    print(f"[probe] device={torch.cuda.get_device_name(0)}")
    # bf16 attention through fp accumulation: a few e-3 abs diff vs fp32 ref is
    # expected; the discriminating signal is strided_vs_packed (same kernel, same
    # numerics) which must be ~0 if strides are honoured.
    tol_kernel = 5e-3   # strided vs packed (same kernel path)
    tol_ref = 5e-2      # vs fp32 manual reference (bf16 rounding)

    # FA3 is Hopper-only (sm90); forcing it elsewhere aborts the process via an
    # uncatchable CUDA error (TMA descriptor), so gate it by device capability.
    major = torch.cuda.get_device_capability(0)[0]
    versions = (2, 3) if major >= 9 else (2,)
    if major < 9:
        print("[probe] device < sm90: testing FA2 only "
              "(FA3/Hopper TMA strided-view support untested here -- see Q-B1').")

    print("\n[probe] WRITE (column-offset view, stock reshape_and_cache):")
    w = run_write_probe()
    print(f"          target_col_diff={w['target_col_diff']:.2e} "
          f"other_cols_diff={w['other_cols_diff']:.2e} "
          f"-> {'OK' if max(w.values()) == 0 else 'FAIL'}")

    any_fail = max(w.values()) != 0
    print("\n[probe] READ (column-offset view, stock flash_attn_varlen_func):")
    for fa_version in versions:
        try:
            r = run_probe(fa_version)
        except Exception as exc:  # noqa: BLE001 -- probe reports, does not crash
            print(f"[probe] fa_version={fa_version}: ERROR {exc!r}")
            any_fail = True
            continue
        verdict = (
            "HONORS STRIDES (cross-layer read feasible)"
            if r["strided_vs_packed"] <= tol_kernel
            and r["strided_vs_ref"] <= tol_ref
            else "ASSUMES PACKED (custom kernel / fallback needed)")
        print(f"[probe] fa_version={fa_version}: {verdict}")
        print(f"          key_view shape={r['key_view_shape']} "
              f"strides={r['key_view_strides']}")
        print(f"          packed_vs_ref   = {r['packed_vs_ref']:.2e}")
        print(f"          strided_vs_ref  = {r['strided_vs_ref']:.2e}")
        print(f"          strided_vs_packed = {r['strided_vs_packed']:.2e} "
              f"(tol {tol_kernel:.0e})")
        if r["strided_vs_packed"] > tol_kernel:
            any_fail = True
    return 2 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
