# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behaviour tests for the eviction gather / write-back.

``_gather_and_writeback_kept_kv`` decides which cached KV survives a chunk
boundary and where it lands. A wrong index here does not crash -- it moves
another head's KV into this one's slots -- and it had no tests, so any change
to the indexing was unverifiable.

These pin what the call observably does: the positions it reports keeping, the
KV those positions carry, where in the pages it writes them, and that the
trailing partial block is zeroed. Columns are given DIFFERENT keep positions
throughout, because index algebra that confuses the column axis with the token
axis still looks right when every column keeps the same thing.

Everything runs on CPU with no model. Run without the root conftest, which
imports a package this fork does not install:

    python -m pytest --noconftest -q tests/v1/attention/test_eviction_writeback.py
"""
import torch

from vllm.v1.attention.compression.executor import CompressionExecutor

BLOCK = 4
PG = 2               # page_group_size
HEAD = 3
POOL = 12            # blocks in the pool
DEV = torch.device("cpu")


def make_executor() -> CompressionExecutor:
    return CompressionExecutor(
        num_layers=1,
        num_kv_heads_per_layer=PG,
        page_group_size=PG,
        head_size=HEAD,
        block_size=BLOCK,
    )


def make_cache() -> torch.Tensor:
    """``[2, POOL, PG, BLOCK, HEAD]`` with every element distinct, so a
    misplaced value is identifiable rather than coincidentally equal."""
    n = 2 * POOL * PG * BLOCK * HEAD
    return torch.arange(n, dtype=torch.float32).reshape(2, POOL, PG, BLOCK, HEAD)


def cached(kv: torch.Tensor, block_ids: torch.Tensor, col: int, pos: int):
    """The KV a token position holds, read the slow, obvious way."""
    return kv[:, block_ids[pos // BLOCK], col, pos % BLOCK]


def call(ex, kv, block_ids, *, sink_size, locked, k_aligned, kept_lo,
         sorted_idx_group, tail_size, tail_lo, kept_length):
    return ex._gather_and_writeback_kept_kv(
        kv_cache=kv,
        block_ids=block_ids,
        sink_idx=torch.arange(sink_size, dtype=torch.long),
        locked=locked,
        k_aligned=k_aligned,
        kept_lo=kept_lo,
        sorted_idx_group=sorted_idx_group,
        tail_idx=torch.arange(tail_size, dtype=torch.long),
        tail_lo=tail_lo,
        kept_length=kept_length,
        device=DEV,
    )


# --- which positions survive ----------------------------------------------


def test_kept_positions_are_sink_then_locked_then_mid_then_tail():
    """The four spans in order. Only the middle differs per column: sink,
    locked and tail are the same positions for every head in the group."""
    ex, kv = make_executor(), make_cache()
    block_ids = torch.tensor([7, 2, 9, 4], dtype=torch.long)  # 16 positions
    # Column 0 promotes offsets 1 and 3, column 1 promotes 0 and 2.
    sorted_idx = torch.tensor([[1, 3, 0, 2], [0, 2, 1, 3]], dtype=torch.long)

    keep = call(ex, kv, block_ids, sink_size=2, locked=1, k_aligned=2,
                kept_lo=3, sorted_idx_group=sorted_idx, tail_size=3,
                tail_lo=13, kept_length=8)

    # sink [0,1] + locked [2] + (sorted mid + kept_lo) + tail [13,14,15]
    assert keep.shape == (PG, 8)
    torch.testing.assert_close(keep[0], torch.tensor([0, 1, 2, 4, 6, 13, 14, 15]))
    torch.testing.assert_close(keep[1], torch.tensor([0, 1, 2, 3, 5, 13, 14, 15]))


def test_the_middle_span_is_sorted_before_it_is_offset():
    """``sorted_idx_group`` arrives in score order; the kept run has to come
    out in position order or the write-back reverses time."""
    ex, kv = make_executor(), make_cache()
    block_ids = torch.tensor([1, 5], dtype=torch.long)
    sorted_idx = torch.tensor([[3, 0, 1, 2], [2, 3, 0, 1]], dtype=torch.long)

    keep = call(ex, kv, block_ids, sink_size=1, locked=0, k_aligned=3,
                kept_lo=1, sorted_idx_group=sorted_idx, tail_size=0,
                tail_lo=8, kept_length=4)

    torch.testing.assert_close(keep[0], torch.tensor([0, 1, 2, 4]))
    torch.testing.assert_close(keep[1], torch.tensor([0, 1, 3, 4]))


def test_no_middle_span_never_touches_sorted_idx():
    """``k_aligned == 0`` is the warm-up chunk, where nothing has been scored
    yet and the caller may pass no indices at all."""
    ex, kv = make_executor(), make_cache()
    block_ids = torch.tensor([3, 8], dtype=torch.long)

    keep = call(ex, kv, block_ids, sink_size=2, locked=0, k_aligned=0,
                kept_lo=2, sorted_idx_group=None, tail_size=2, tail_lo=6,
                kept_length=4)

    torch.testing.assert_close(keep[0], torch.tensor([0, 1, 6, 7]))
    torch.testing.assert_close(keep[1], torch.tensor([0, 1, 6, 7]))


# --- what lands in the pages ----------------------------------------------


def test_the_kept_kv_moves_to_the_front_of_the_same_pages():
    """Each column's surviving KV is compacted to slots ``[0, kept_length)``
    of the cluster's own pages, column by column."""
    ex, kv = make_executor(), make_cache()
    block_ids = torch.tensor([7, 2, 9, 4], dtype=torch.long)
    sorted_idx = torch.tensor([[1, 3, 0, 2], [0, 2, 1, 3]], dtype=torch.long)
    before = kv.clone()

    keep = call(ex, kv, block_ids, sink_size=2, locked=1, k_aligned=2,
                kept_lo=3, sorted_idx_group=sorted_idx, tail_size=3,
                tail_lo=13, kept_length=8)

    for col in range(PG):
        for new_pos, old_pos in enumerate(keep[col].tolist()):
            torch.testing.assert_close(
                cached(kv, block_ids, col, new_pos),
                cached(before, block_ids, col, old_pos),
                msg=f"column {col} slot {new_pos} did not take position {old_pos}",
            )


def test_pages_past_the_written_blocks_are_untouched():
    ex, kv = make_executor(), make_cache()
    block_ids = torch.tensor([7, 2, 9, 4], dtype=torch.long)
    sorted_idx = torch.tensor([[1, 3, 0, 2], [0, 2, 1, 3]], dtype=torch.long)
    before = kv.clone()

    call(ex, kv, block_ids, sink_size=2, locked=1, k_aligned=2, kept_lo=3,
         sorted_idx_group=sorted_idx, tail_size=3, tail_lo=13, kept_length=8)

    # kept_length 8 fills blocks 0 and 1 of the cluster; 9 and 4 are untouched.
    torch.testing.assert_close(kv[:, 9], before[:, 9])
    torch.testing.assert_close(kv[:, 4], before[:, 4])
    # And so is every page outside the cluster.
    outside = [b for b in range(POOL) if b not in block_ids.tolist()]
    torch.testing.assert_close(kv[:, outside], before[:, outside])


def test_a_trailing_partial_block_is_zero_padded():
    """The write-back is block-aligned, so a kept length that stops mid-block
    must leave zeros behind rather than stale KV a later read could pick up."""
    ex, kv = make_executor(), make_cache()
    block_ids = torch.tensor([1, 5, 10], dtype=torch.long)
    sorted_idx = torch.tensor([[2, 0, 1], [1, 2, 0]], dtype=torch.long)

    call(ex, kv, block_ids, sink_size=1, locked=0, k_aligned=2, kept_lo=1,
         sorted_idx_group=sorted_idx, tail_size=2, tail_lo=10, kept_length=5)

    # 5 kept -> 2 blocks written, slots 5..7 of the second block are padding.
    for pos in (5, 6, 7):
        torch.testing.assert_close(
            cached(kv, block_ids, 0, pos), torch.zeros(2, HEAD))
        torch.testing.assert_close(
            cached(kv, block_ids, 1, pos), torch.zeros(2, HEAD))


def test_scattered_pages_are_addressed_through_the_block_table():
    """The cluster's pages are not contiguous in the pool, so the gather has
    to resolve every position through ``block_ids``."""
    ex, kv = make_executor(), make_cache()
    block_ids = torch.tensor([11, 0, 6], dtype=torch.long)
    sorted_idx = torch.tensor([[2, 1, 0], [0, 1, 2]], dtype=torch.long)
    before = kv.clone()

    keep = call(ex, kv, block_ids, sink_size=1, locked=0, k_aligned=2,
                kept_lo=1, sorted_idx_group=sorted_idx, tail_size=1,
                tail_lo=11, kept_length=4)

    for col in range(PG):
        for new_pos, old_pos in enumerate(keep[col].tolist()):
            torch.testing.assert_close(
                cached(kv, block_ids, col, new_pos),
                cached(before, block_ids, col, old_pos))
