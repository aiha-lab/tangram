# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How many blocks a ragged request costs.

Compression leaves each head group a different length, so what a request
occupies is about the SUM over groups::

    blocks ~= sum_g ceil(len_g / block_size)

Sizing it as a rectangle over the longest group instead -- ``ceil(max_g len_g /
block_size) x num_groups`` -- hands the pages back that compression just freed,
which is invisible in an accuracy benchmark and shows up only as fewer
concurrent requests. These pin the sum, and that decode still allocates once
per block of tokens rather than on almost every step.

Everything runs on CPU with no model. Run without the root conftest, which
imports a package this fork does not install:

    python -m pytest --noconftest -q tests/v1/core/test_ragged_allocation.py
"""
import numpy as np
import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.ragged_allocation import (
    group_block_depths,
    spare_blocks,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, RaggedAttentionSpec

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 4
PAGE_GROUP_SIZE = 2
NUM_KV_HEADS = 4
GROUPS_PER_LAYER = NUM_KV_HEADS // PAGE_GROUP_SIZE
NUM_LAYERS = 2
NUM_HEAD_GROUPS = GROUPS_PER_LAYER * NUM_LAYERS
POOL_BLOCKS = 1024

REQ = "req-0"


def make_manager() -> SingleTypeKVCacheManager:
    spec = RaggedAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=8,
        dtype=torch.float16,
        page_group_size=PAGE_GROUP_SIZE,
    )
    pool = BlockPool(
        POOL_BLOCKS,
        enable_caching=False,
        ragged=True,
        num_head_groups=NUM_HEAD_GROUPS,
    )
    # ``spec_manager_map`` routes RaggedAttentionSpec here.
    return FullAttentionManager(spec, pool, kv_cache_group_id=0)


def exact_blocks(per_group_tokens) -> int:
    """The exact sum rule, written out for the tests to compare against."""
    return int(np.ceil(np.asarray(per_group_tokens) / BLOCK_SIZE).sum())


def held(manager: SingleTypeKVCacheManager) -> int:
    ids = manager.req_to_block_ids.get(REQ)
    return 0 if ids is None else int(ids.size)


# --- the scalar path: before compression every group is the same length ---


def test_a_scalar_need_costs_one_rectangle():
    """Pre-compression all groups hold the same tokens, so the rectangle and
    the sum agree; the scalar call must keep working unchanged."""
    manager = make_manager()

    cost = manager.get_num_blocks_to_allocate(REQ, 10, [])

    assert cost == 3 * NUM_HEAD_GROUPS  # ceil(10 / 4) = 3 per group


def test_a_scalar_need_hands_out_that_many_ids():
    manager = make_manager()

    new_ids = manager.allocate_new_blocks(REQ, 10)

    assert new_ids.size == 3 * NUM_HEAD_GROUPS
    assert held(manager) == 3 * NUM_HEAD_GROUPS


# --- the compressed path: after compression the groups diverge ----------

# Kept lengths a compression could leave: every group at a different offset
# inside its last block, so the groups cross block edges on different steps.
KEPT = np.array([12, 5, 9, 6], dtype=np.int64)
LONGEST = int(KEPT.max())
MAX_LEN = 1 << 20


def compact(manager: SingleTypeKVCacheManager) -> None:
    """Hold what compaction leaves: each group trimmed to its own last block."""
    manager.allocate_new_blocks(REQ, LONGEST, spare_blocks(KEPT, BLOCK_SIZE))
    assert held(manager) == exact_blocks(KEPT)


def test_spare_blocks_counts_what_short_groups_need_less():
    """Gaps to the longest group are 0, 7, 3 and 6 tokens: whole blocks 0, 1,
    0 and 1."""
    assert spare_blocks(KEPT, BLOCK_SIZE) == 2


def test_spare_blocks_is_zero_when_the_groups_agree():
    assert spare_blocks(np.full(NUM_HEAD_GROUPS, 10), BLOCK_SIZE) == 0


def test_a_compressed_need_costs_the_sum_not_the_rectangle():
    """The whole point: groups that kept 12, 5, 9 and 6 tokens are priced at
    3 + 2 + 3 + 2 = 10 blocks, not at 3 x 4 = 12."""
    manager = make_manager()

    cost = manager.get_num_blocks_to_allocate(
        REQ, LONGEST, [], spare_blocks(KEPT, BLOCK_SIZE))

    assert cost == exact_blocks(KEPT) == 10


def test_the_pool_only_loses_the_sum():
    """The blocks really leave the pool, so the rule decides how many other
    requests fit."""
    manager = make_manager()
    free_before = manager.block_pool.num_free_blocks

    compact(manager)

    assert manager.block_pool.num_free_blocks == free_before - 10


def test_no_spare_blocks_matches_the_scalar_rule():
    """Before the first compression nothing is spared, which is what makes
    one code path serve both."""
    manager = make_manager()

    assert (manager.get_num_blocks_to_allocate(REQ, 10, [], 0)
            == manager.get_num_blocks_to_allocate(REQ, 10, []))


def test_decode_allocates_once_per_block_of_tokens():
    """Groups top up together at the longest group's block edge. Topping each
    up at its own edge allocates on almost every decode step, because the
    edges are staggered; that per-step work is what this rule removes."""
    manager = make_manager()
    compact(manager)
    spare = spare_blocks(KEPT, BLOCK_SIZE)
    steps = 4 * BLOCK_SIZE

    allocations = 0
    for step in range(1, steps + 1):
        need = LONGEST + step
        if manager.get_num_blocks_to_allocate(REQ, need, [], spare) == 0:
            continue
        manager.allocate_new_blocks(REQ, need, spare)
        allocations += 1

    assert allocations == steps // BLOCK_SIZE


def test_decode_holds_at_most_one_block_per_group_over_the_sum():
    """Topping up together reserves at most the rest of one block per group
    ahead of time, so the saving compression made survives decode."""
    manager = make_manager()
    compact(manager)
    spare = spare_blocks(KEPT, BLOCK_SIZE)

    for step in range(1, 4 * BLOCK_SIZE + 1):
        manager.allocate_new_blocks(REQ, LONGEST + step, spare)
        exact = exact_blocks(KEPT + step)
        assert exact <= held(manager) <= exact + NUM_HEAD_GROUPS


# --- the contract the worker has to satisfy -------------------------------


@pytest.mark.parametrize("new_tokens", [0, 1, 3, 4, 7, 16])
def test_the_worker_depths_add_up_to_the_price(new_tokens):
    """The block table fills each group to ``group_block_depths``; the
    allocator hands over ids for ``cdiv(longest + new) x groups - spare``.
    The two must agree, or the block table is left with ids it cannot place
    or groups it cannot fill."""
    manager = make_manager()
    compact(manager)
    spare = spare_blocks(KEPT, BLOCK_SIZE)

    handed = manager.allocate_new_blocks(REQ, LONGEST + new_tokens, spare)
    depths = group_block_depths(KEPT, new_tokens, MAX_LEN, BLOCK_SIZE)

    held_before = np.ceil(KEPT / BLOCK_SIZE).astype(np.int64)
    assert handed.size == int((depths - held_before).sum())


@pytest.mark.parametrize("new_tokens", [0, 1, 3, 4, 7, 16])
def test_every_group_has_room_for_its_tokens(new_tokens):
    depths = group_block_depths(KEPT, new_tokens, MAX_LEN, BLOCK_SIZE)

    room = depths * BLOCK_SIZE

    assert (room >= KEPT + new_tokens).all()


# --- the base path: no head groups, nothing to spare ----------------------


def test_the_base_path_prices_what_it_allocates():
    """``spare_blocks`` is a ragged quantity. The base path must ignore it in
    the price as it does in the allocation, or it admits a request on fewer
    blocks than it then takes."""
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=8,
        dtype=torch.float16,
    )
    pool = BlockPool(POOL_BLOCKS, enable_caching=False)
    manager = FullAttentionManager(spec, pool, kv_cache_group_id=0)

    cost = manager.get_num_blocks_to_allocate(REQ, 10, [], spare_blocks=2)
    new_blocks = manager.allocate_new_blocks(REQ, 10, spare_blocks=2)

    assert cost == len(new_blocks) == 3
