# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the slot score sources (budget-regime score provenance).

CPU only: reading cached keys and scoring them is layout arithmetic plus a
cosine, so a synthetic cache written through the documented column-major layout
is enough — and pins that layout against the helper both the writeback and the
rescoring path use.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm.v1.attention.compression.keydiff import KeyDiffScorer
from vllm.v1.attention.compression.slot_scores import (
    KVCacheView,
    PersistedChunkScores,
    RecomputedCacheScores,
    make_slot_score_source,
)
from vllm.v1.attention.compression.snapkv import SnapKVScorer

NUM_LAYERS = 2
NUM_KV_HEADS = 4
PAGE_GROUP_SIZE = 2
NUM_GROUPS = NUM_KV_HEADS // PAGE_GROUP_SIZE
BLOCK_SIZE = 4
HEAD_SIZE = 8
NUM_BLOCKS = 32


def build_cache_and_view(
    live_lens: np.ndarray,
    generator: torch.Generator,
) -> tuple[KVCacheView, dict[tuple[int, int], torch.Tensor]]:
    """Lay out random keys for every (layer, group) and return a view plus the
    keys that were written, addressed by (layer, group).

    Writes through the documented column-major layout ``[2, num_blocks,
    page_group_size, block_size, head_size]``: slot ``t`` of column ``c`` lives
    at block ``t // block_size``, offset ``t % block_size``.
    """
    caches = [
        torch.zeros(2, NUM_BLOCKS, PAGE_GROUP_SIZE, BLOCK_SIZE, HEAD_SIZE)
        for _ in range(NUM_LAYERS)
    ]
    max_blocks = NUM_BLOCKS
    block_table = torch.zeros(
        1, NUM_LAYERS * NUM_GROUPS, max_blocks, dtype=torch.int32)
    written: dict[tuple[int, int], torch.Tensor] = {}
    next_block = 0
    for layer_idx in range(NUM_LAYERS):
        for group_idx in range(NUM_GROUPS):
            num_positions = int(live_lens[layer_idx, group_idx])
            num_blocks = (num_positions + BLOCK_SIZE - 1) // BLOCK_SIZE
            block_ids = list(range(next_block, next_block + num_blocks))
            next_block += num_blocks
            row = layer_idx * NUM_GROUPS + group_idx
            for slot, block_id in enumerate(block_ids):
                block_table[0, row, slot] = block_id
            keys = torch.rand(
                PAGE_GROUP_SIZE, num_positions, HEAD_SIZE,
                generator=generator)
            for col in range(PAGE_GROUP_SIZE):
                for pos in range(num_positions):
                    caches[layer_idx][
                        0, block_ids[pos // BLOCK_SIZE], col,
                        pos % BLOCK_SIZE] = keys[col, pos]
            written[(layer_idx, group_idx)] = keys
    view = KVCacheView(
        layer_kv_caches=caches,
        block_table_gpu=block_table,
        row_idx=0,
        compressed_layer_ids=np.arange(NUM_LAYERS),
        num_groups=NUM_GROUPS,
        block_size=BLOCK_SIZE,
    )
    return view, written


def test_cache_view_reads_the_documented_layout():
    """``cluster_keys`` must return exactly the keys that were written, in slot
    order — the whole rescoring path rests on this addressing."""
    generator = torch.Generator().manual_seed(0)
    live_lens = np.array([[7, 4], [12, 1]], dtype=np.int64)
    view, written = build_cache_and_view(live_lens, generator)
    for layer_idx in range(NUM_LAYERS):
        for group_idx in range(NUM_GROUPS):
            num_positions = int(live_lens[layer_idx, group_idx])
            got = view.cluster_keys(layer_idx, group_idx, num_positions)
            assert got.shape == (
                PAGE_GROUP_SIZE, num_positions, HEAD_SIZE)
            torch.testing.assert_close(got, written[(layer_idx, group_idx)])


def test_keydiff_cached_score_is_the_paper_formula():
    """``score_cached_keys`` is Eq. (8): the anchor is the mean direction of ALL
    the keys handed to it, and every key is scored against it."""
    generator = torch.Generator().manual_seed(1)
    keys = torch.rand(PAGE_GROUP_SIZE, 13, HEAD_SIZE, generator=generator)
    scorer = KeyDiffScorer(num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE)

    got = scorer.score_cached_keys(keys)
    anchor = F.normalize(keys.float(), p=2, dim=-1).mean(dim=1, keepdim=True)
    expected = -F.cosine_similarity(keys.float(), anchor, dim=-1)
    torch.testing.assert_close(got, expected)
    assert got.shape == (PAGE_GROUP_SIZE, 13)


def test_keydiff_whole_cache_anchor_differs_from_the_chunk_anchor():
    """The change the budget regime needs is real: scoring against the whole
    cache's anchor ranks positions differently from scoring each chunk against
    its own, which is why stored per-chunk scores are not interchangeable."""
    generator = torch.Generator().manual_seed(2)
    scorer = KeyDiffScorer(num_kv_heads=1, head_size=HEAD_SIZE)
    # Two chunks with clearly different key distributions.
    chunk_a = torch.rand(1, 16, HEAD_SIZE, generator=generator)
    chunk_b = torch.rand(1, 16, HEAD_SIZE, generator=generator) + 3.0
    whole = torch.cat([chunk_a, chunk_b], dim=1)

    per_chunk = torch.cat(
        [scorer.score_cached_keys(chunk_a),
         scorer.score_cached_keys(chunk_b)], dim=1)
    whole_cache = scorer.score_cached_keys(whole)
    per_chunk_rank = per_chunk.argsort(dim=-1)
    whole_rank = whole_cache.argsort(dim=-1)
    assert not torch.equal(per_chunk_rank, whole_rank), (
        "if these agreed, recomputing over the cache would be pointless")


def test_recompute_source_fills_every_live_slot():
    """The source must leave a score at every live slot and nothing selectable
    beyond it, for each (layer, group) independently."""
    from vllm.v1.attention.compression.slot_scores import (
        ChunkScoreInputs,
        SlotFillTarget,
    )
    generator = torch.Generator().manual_seed(3)
    chunk_len = 4
    prev_lens = np.array([[3, 0], [8, 1]], dtype=np.int64)
    live_lens = prev_lens + chunk_len
    view, written = build_cache_and_view(live_lens, generator)

    capacity = 32
    buffer = torch.full(
        (NUM_LAYERS, NUM_KV_HEADS, capacity), float("-inf"))
    flat = buffer.view(NUM_LAYERS * NUM_KV_HEADS, capacity)
    # Identity maps: member row m = layer * num_kv_heads + head belongs to
    # cluster layer * num_groups + head // page_group_size at column
    # head % page_group_size.
    member_to_cluster = torch.tensor([
        layer * NUM_GROUPS + head // PAGE_GROUP_SIZE
        for layer in range(NUM_LAYERS) for head in range(NUM_KV_HEADS)])
    cluster_members = np.array([
        [layer * NUM_KV_HEADS + group * PAGE_GROUP_SIZE + col
         for col in range(PAGE_GROUP_SIZE)]
        for layer in range(NUM_LAYERS) for group in range(NUM_GROUPS)])

    scorer = KeyDiffScorer(num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE)
    source = RecomputedCacheScores(scorer)
    source.fill(
        SlotFillTarget(
            buffer=buffer, flat=flat, member_to_cluster=member_to_cluster,
            cluster_members_cpu=cluster_members, num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS, num_groups=NUM_GROUPS,
            neg_inf=float("-inf")),
        ChunkScoreInputs(
            pending=torch.empty(NUM_LAYERS, NUM_KV_HEADS, 0),
            prev_lens_cpu=prev_lens,
            prev_lens_device=torch.from_numpy(prev_lens),
            chunk_len=chunk_len,
            cache_view=view),
    )

    for layer_idx in range(NUM_LAYERS):
        for group_idx in range(NUM_GROUPS):
            num_positions = int(live_lens[layer_idx, group_idx])
            expected = scorer.score_cached_keys(
                written[(layer_idx, group_idx)])
            for col in range(PAGE_GROUP_SIZE):
                head = group_idx * PAGE_GROUP_SIZE + col
                torch.testing.assert_close(
                    buffer[layer_idx, head, :num_positions], expected[col])
                assert torch.all(torch.isinf(
                    buffer[layer_idx, head, num_positions:])), (
                    "slots past the live extent must stay unselectable")


def test_recompute_source_requires_cache_access():
    from vllm.v1.attention.compression.slot_scores import (
        ChunkScoreInputs,
        SlotFillTarget,
    )
    source = RecomputedCacheScores(
        KeyDiffScorer(num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE))
    target = SlotFillTarget(
        buffer=torch.zeros(1, 1, 1), flat=torch.zeros(1, 1),
        member_to_cluster=torch.zeros(1, dtype=torch.long),
        cluster_members_cpu=np.zeros((1, 1), dtype=np.int64),
        num_layers=1, num_kv_heads=1, num_groups=1, neg_inf=0.0)
    with pytest.raises(RuntimeError, match="KVCacheView"):
        source.fill(target, ChunkScoreInputs(
            pending=torch.zeros(1, 1, 0),
            prev_lens_cpu=np.zeros((1, 1), dtype=np.int64),
            prev_lens_device=torch.zeros(1, 1, dtype=torch.long),
            chunk_len=0, cache_view=None))


def test_source_selection_follows_the_scorer():
    """A scorer that can rescore the cache gets the recompute source; one that
    cannot keeps its chunk scores. Not a user-facing choice."""
    keydiff = KeyDiffScorer(num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE)
    snapkv = SnapKVScorer(
        num_kv_heads=NUM_KV_HEADS, num_q_per_kv=2, head_size=HEAD_SIZE,
        snap_window=8, snap_kernel=3)
    assert isinstance(make_slot_score_source(keydiff), RecomputedCacheScores)
    assert isinstance(make_slot_score_source(snapkv), PersistedChunkScores)
    assert isinstance(make_slot_score_source(None), PersistedChunkScores)
    # A recomputing source makes the per-chunk scorer unnecessary; a persisting
    # one depends on it.
    assert not RecomputedCacheScores(keydiff).needs_chunk_scores
    assert PersistedChunkScores().needs_chunk_scores


def test_budget_regime_skips_chunk_scoring_only_when_recomputing():
    from vllm.v1.attention.compression.eviction_regime import (
        BudgetRegime,
        RatioRegime,
    )
    keydiff = KeyDiffScorer(num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE)
    recompute = RecomputedCacheScores(keydiff)
    persist = PersistedChunkScores()
    assert not BudgetRegime().consumes_chunk_scores(recompute)
    assert BudgetRegime().consumes_chunk_scores(persist)
    # The ratio regime has only the chunk's own scores to rank, whatever the
    # scorer could do.
    assert RatioRegime().consumes_chunk_scores(recompute)
    assert not RatioRegime().uses_slot_scores
    assert BudgetRegime().uses_slot_scores
