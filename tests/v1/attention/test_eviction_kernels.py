# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The batched Triton write-back must equal the torch reference bit for bit.

Both backends receive one :class:`EvictionPlan` over several layers and
head-groups with DIFFERENT kept lengths, locked prefixes, top-k counts and
tail offsets, on clones of the same cache and, when a score buffer follows
the positions, of the same buffer; afterwards caches and buffer must be
identical. The top-k positions differ per column so an error that confuses
the column axis with the token axis cannot pass.

Needs a GPU. Run without the root conftest:

    python -m pytest --noconftest -q tests/v1/attention/test_eviction_kernels.py
"""
import numpy as np
import pytest
import torch

import vllm.v1.attention.compression.eviction_writeback as writeback
from vllm.v1.attention.compression.eviction_writeback import (
    NUM_PLAN_COLS,
    EvictionPlan,
    PlanCol,
    SlotCompactionTarget,
    TorchWriteback,
    TritonWriteback,
    gather_and_writeback_kept_kv,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels need a GPU")

DEV = torch.device("cuda")
BLOCK = 16


def make_case(*, seed, num_layers, num_groups, pg, head, chunk, sink, tail,
              dtype, locked_regime):
    """One boundary over ``num_layers x num_groups`` clusters.

    Each cluster gets its own total_seen, locked prefix and kept length. With
    ``locked_regime`` the locked prefix is non-zero (ratio regime); without it
    the eval region starts right after the sink (budget regime).
    """
    rng = np.random.default_rng(seed)
    num_clusters = num_layers * num_groups
    eval_cap = 2 * chunk
    max_blocks = (2 * chunk + BLOCK - 1) // BLOCK + 1
    pool = num_clusters * max_blocks
    kv = [torch.randn(2, pool, pg, BLOCK, head, dtype=dtype, device=DEV)
          for _ in range(num_layers)]

    # Every cluster owns disjoint random pages of the pool.
    perm = torch.randperm(pool, device=DEV).to(torch.int32)
    block_table = perm.view(1, num_clusters, max_blocks).clone()

    sorted_idx = torch.zeros(num_clusters, pg, eval_cap, dtype=torch.long,
                             device=DEV)
    rows = []
    for c in range(num_clusters):
        if locked_regime:
            # A first chunk (prev 0) scores past its own sink and window;
            # a later one has prev >= sink + tail and scores the whole chunk.
            first = rng.random() < 0.25
            prev = 0 if first else int(rng.integers(sink + tail, chunk))
            locked = 0 if first else prev - sink - tail
            eval_len = chunk - sink - tail if first else chunk
        else:
            prev = int(rng.integers(0, chunk // 2))
            locked = 0
            eval_len = prev + chunk - sink - tail
        total = prev + chunk
        kept_lo = sink + locked
        tail_lo = total - tail
        assert kept_lo + eval_len == tail_lo
        k = int(rng.integers(0, eval_len + 1))
        kept = kept_lo + k + tail
        # Occasionally keep nothing selected, or a block-aligned length.
        if rng.random() < 0.2:
            k = 0
            kept = kept_lo + tail
        for col in range(pg):
            sorted_idx[c, col, :eval_len] = torch.randperm(
                eval_len, device=DEV)
        layer, group = divmod(c, num_groups)
        rows.append([layer, c * max_blocks, c, kept_lo, k, tail_lo, kept])
    table = np.array(rows, dtype=np.int64).reshape(-1, NUM_PLAN_COLS)
    plan = EvictionPlan(table=table, sink_size=sink, tail_size=tail,
                        eval_len=int(max(r[PlanCol.TAIL_LO] - r[PlanCol.KEPT_LO]
                                         for r in rows)))
    return kv, block_table, sorted_idx, plan


def make_scores(num_clusters, pg, capacity, dtype, *, shuffle):
    """A score buffer with one row per member. ``shuffle`` permutes which
    member row each (cluster, column) owns, as a real cluster map does, and
    leaves one cluster's worth of rows outside every plan."""
    rows = num_clusters * pg
    flat = torch.randn(rows + pg, capacity, dtype=dtype, device=DEV)
    order = np.random.default_rng(rows).permutation(rows) if shuffle else (
        np.arange(rows))
    members = order.reshape(num_clusters, pg)
    return flat, members


def run_both(case, pg, head, scores=None):
    kv, block_table, sorted_idx, plan = case
    kv_ref = [t.clone() for t in kv]
    idx_ref = sorted_idx.clone()
    kv_tri = [t.clone() for t in kv]
    idx_tri = sorted_idx.clone()
    scores_ref = scores_tri = None
    if scores is not None:
        flat, members = scores
        neg_inf = float(torch.finfo(flat.dtype).min)
        scores_ref = SlotCompactionTarget(flat.clone(), members, neg_inf)
        scores_tri = SlotCompactionTarget(flat.clone(), members, neg_inf)

    TorchWriteback(BLOCK).run(plan, kv_ref, block_table, idx_ref, scores_ref)
    mask = torch.zeros_like(idx_tri, dtype=torch.uint8)
    tri = TritonWriteback(BLOCK, pg, head, mask)
    tri.run(plan, kv_tri, block_table, idx_tri, scores_tri)
    torch.cuda.synchronize()
    return kv_ref, kv_tri, scores_ref, scores_tri, plan


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("pg,head", [(4, 128), (2, 64), (1, 128), (8, 96)])
@pytest.mark.parametrize("locked_regime", [True, False])
def test_triton_matches_reference(dtype, pg, head, locked_regime):
    case = make_case(seed=pg * 100 + head + int(locked_regime),
                     num_layers=3, num_groups=2, pg=pg, head=head, chunk=512,
                     sink=4, tail=32, dtype=dtype, locked_regime=locked_regime)
    kv_ref, kv_tri, _, _, plan = run_both(case, pg, head)

    for layer, (a, b) in enumerate(zip(kv_ref, kv_tri)):
        assert torch.equal(a, b), f"layer {layer} KV differs"


@pytest.mark.parametrize("score_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shuffle", [False, True])
def test_score_buffer_follows_the_same_positions(score_dtype, shuffle):
    """Budget regime: the per-slot scores move with their KV and every slot
    from ``kept`` on is blanked; rows no plan touches stay as they were."""
    pg, head, num_layers, num_groups = 4, 64, 3, 2
    case = make_case(seed=21, num_layers=num_layers, num_groups=num_groups,
                     pg=pg, head=head, chunk=512, sink=4, tail=32,
                     dtype=torch.bfloat16, locked_regime=False)
    scores = make_scores(num_layers * num_groups, pg, capacity=2 * 512 + 64,
                         dtype=score_dtype, shuffle=shuffle)
    kv_ref, kv_tri, s_ref, s_tri, plan = run_both(case, pg, head, scores)

    for a, b in zip(kv_ref, kv_tri):
        assert torch.equal(a, b)
    assert torch.equal(s_ref.flat, s_tri.flat)
    # The reference blanked from ``kept``; check the kernel path did the same
    # and did not touch the spare rows.
    flat0, members = scores
    spare = torch.arange(members.size, members.size + pg, device=DEV)
    assert torch.equal(s_tri.flat[spare], flat0[spare])
    for row, kept in zip(plan.table, plan.table[:, PlanCol.KEPT]):
        for member in members[int(row[PlanCol.CLUSTER])]:
            assert bool((s_tri.flat[member, int(kept):] == s_tri.neg_inf).all())


def test_positions_are_strictly_increasing():
    """The in-place kernel's safety argument: within a column the kept
    positions ascend, so slot j's source is at or past j."""
    case = make_case(seed=7, num_layers=2, num_groups=2, pg=4, head=64,
                     chunk=256, sink=2, tail=8, dtype=torch.bfloat16,
                     locked_regime=True)
    kv, block_table, sorted_idx, plan = case
    bt_flat = block_table.reshape(-1)
    for row in plan.table:
        kept_lo, k, tail_lo, kept = (int(row[c]) for c in (
            PlanCol.KEPT_LO, PlanCol.K_ALIGNED, PlanCol.TAIL_LO, PlanCol.KEPT))
        n_blocks = (tail_lo + plan.tail_size + BLOCK - 1) // BLOCK
        off = int(row[PlanCol.BT_OFFSET])
        pos = gather_and_writeback_kept_kv(
            kv_cache=kv[int(row[PlanCol.LAYER])].clone(),
            block_ids=bt_flat[off:off + n_blocks].long(), block_size=BLOCK,
            sink_idx=torch.arange(plan.sink_size, device=DEV),
            locked=kept_lo - plan.sink_size, k_aligned=k, kept_lo=kept_lo,
            sorted_idx_group=sorted_idx[int(row[PlanCol.CLUSTER])],
            tail_idx=torch.arange(plan.tail_size, device=DEV),
            tail_lo=tail_lo, kept_length=kept)
        if pos.shape[1] == 0:
            continue
        assert bool((pos[:, 1:] > pos[:, :-1]).all())
        j = torch.arange(pos.shape[1], device=pos.device)
        assert bool((pos >= j).all())


def test_no_sink_no_tail():
    """Degenerate geometry: the middle span starts at slot 0 and nothing is
    pinned behind it."""
    case = make_case(seed=3, num_layers=1, num_groups=1, pg=2, head=32,
                     chunk=64, sink=0, tail=0, dtype=torch.float16,
                     locked_regime=False)
    kv_ref, kv_tri, _, _, _ = run_both(case, 2, 32)
    assert torch.equal(kv_ref[0], kv_tri[0])


def _compiled_variants() -> int:
    kernels = [writeback._sort_kept_positions_kernel,
               writeback._writeback_prefix_kernel,
               writeback._writeback_parallel_kernel,
               writeback._writeback_pad_kernel]
    return sum(len(cache) for k in kernels
               for cache, *_ in k.device_caches.values())


def test_warmup_compiles_every_variant_a_boundary_needs():
    """A boundary's plan differs from the warm-up's in cluster count, eval
    width, tile count and buffer offsets. None of that may trigger a compile:
    the first request would otherwise stall for the JIT, which is exactly what
    the warm-up exists to prevent."""
    pg, head = 4, 128
    case = make_case(seed=11, num_layers=3, num_groups=2, pg=pg, head=head,
                     chunk=512, sink=4, tail=32, dtype=torch.bfloat16,
                     locked_regime=True)
    kv, block_table, sorted_idx, plan = case
    flat, members = make_scores(6, pg, capacity=1088, dtype=torch.float32,
                                shuffle=True)
    scores = SlotCompactionTarget(flat, members, float(torch.finfo(
        torch.float32).min))
    mask = torch.zeros_like(sorted_idx, dtype=torch.uint8)
    tri = TritonWriteback(BLOCK, pg, head, mask)
    tri.warmup(torch.empty(2, 2, pg, BLOCK, head, dtype=torch.bfloat16,
                           device=DEV), score_dtype=torch.float32)
    torch.cuda.synchronize()

    before = _compiled_variants()
    tri.run(plan, kv, block_table, sorted_idx, scores)
    torch.cuda.synchronize()
    assert _compiled_variants() == before
