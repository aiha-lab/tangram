# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How many blocks a ragged request costs.

Compression leaves each head group a different length, so what a request
occupies is the SUM over groups::

    blocks = sum_g ceil(len_g / block_size)

Sizing it as a rectangle over the longest group instead -- ``ceil(max_g len_g /
block_size) x num_groups`` -- hands the pages back that compression just freed,
which is invisible in an accuracy benchmark and shows up only as fewer
concurrent requests. These pin the sum.

Everything runs on CPU with no model. Run without the root conftest, which
imports a package this fork does not install:

    python -m pytest --noconftest -q tests/v1/core/test_ragged_allocation.py
"""
import numpy as np
import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheManager,
)
from vllm.v1.kv_cache_interface import RaggedAttentionSpec

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


def blocks_for(per_group_tokens) -> int:
    """The sum rule, written out for the tests to compare against."""
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


# --- the per-group path: after compression the groups diverge -------------


def test_a_per_group_need_costs_the_sum_not_the_rectangle():
    """The whole point: a request whose groups kept 12, 4, 8 and 4 tokens is
    priced at 3 + 1 + 2 + 1 = 7 blocks, not at 3 x 4 = 12."""
    manager = make_manager()
    per_group = np.array([12, 4, 8, 4], dtype=np.int64)

    cost = manager.get_num_blocks_to_allocate(REQ, per_group, [])

    assert cost == blocks_for(per_group) == 7


def test_a_per_group_need_hands_out_the_sum():
    manager = make_manager()
    per_group = np.array([12, 4, 8, 4], dtype=np.int64)

    new_ids = manager.allocate_new_blocks(REQ, per_group)

    assert new_ids.size == 7
    assert held(manager) == 7


def test_a_uniform_per_group_need_matches_the_scalar_rule():
    """The two rules are the same function when the groups agree, which is
    what makes the switch safe before the first compression."""
    manager = make_manager()

    scalar = manager.get_num_blocks_to_allocate(REQ, 10, [])
    vector = manager.get_num_blocks_to_allocate(
        REQ, np.full(NUM_HEAD_GROUPS, 10, dtype=np.int64), []
    )

    assert scalar == vector


def test_a_per_group_need_subtracts_what_the_request_already_holds():
    manager = make_manager()
    manager.allocate_new_blocks(REQ, np.array([12, 4, 8, 4], dtype=np.int64))

    # Every group grows by one token: only the groups that cross a block
    # boundary cost anything. 12 -> 13 crosses, 4 -> 5 crosses, 8 -> 9
    # crosses, 4 -> 5 crosses.
    grown = np.array([13, 5, 9, 5], dtype=np.int64)
    cost = manager.get_num_blocks_to_allocate(REQ, grown, [])

    assert cost == blocks_for(grown) - 7 == 4


def test_a_group_that_needs_nothing_new_costs_nothing():
    """A decode step after compression must not re-grow the short groups back
    to the longest one -- that is the pages coming straight back."""
    manager = make_manager()
    manager.allocate_new_blocks(REQ, np.array([12, 4, 8, 4], dtype=np.int64))

    # One more token, and no group crosses a block boundary.
    unchanged = np.array([9, 1, 5, 1], dtype=np.int64)
    cost = manager.get_num_blocks_to_allocate(REQ, unchanged, [])

    assert cost == 0


def test_the_pool_only_loses_the_sum():
    """The blocks really leave the pool, so the rule decides how many other
    requests fit."""
    manager = make_manager()
    free_before = manager.block_pool.num_free_blocks

    manager.allocate_new_blocks(REQ, np.array([12, 4, 8, 4], dtype=np.int64))

    assert manager.block_pool.num_free_blocks == free_before - 7


def test_a_per_group_need_rejects_a_wrong_group_count():
    """A vector that is not one entry per head group means the caller and the
    pool disagree about the layout."""
    manager = make_manager()

    with pytest.raises(ValueError, match="num_head_groups"):
        manager.get_num_blocks_to_allocate(
            REQ, np.array([12, 4], dtype=np.int64), []
        )


# --- the contract the worker has to satisfy -------------------------------


@pytest.mark.parametrize(
    "kept,new_tokens",
    [
        ([12, 4, 8, 4], 1),
        ([12, 4, 8, 4], 7),
        ([1, 1, 1, 1], 16),
        ([0, 0, 0, 0], 5),
        ([13, 5, 9, 5], 0),
    ],
)
def test_the_worker_target_consumes_exactly_what_is_handed_over(
    kept, new_tokens
):
    """Both sides derive the per-group depth from the same two inputs: the
    kept lengths and the tokens this step writes. The block table fills each
    group to ``ceil((kept_g + new) / block)``, so the count the allocator
    hands over must be exactly the shortfall summed over groups -- otherwise
    the block table would be left with ids it cannot place."""
    manager = make_manager()
    kept = np.asarray(kept, dtype=np.int64)
    manager.allocate_new_blocks(REQ, kept)
    depth = np.ceil(kept / BLOCK_SIZE).astype(np.int64)

    handed = manager.allocate_new_blocks(REQ, kept + new_tokens)

    target = np.ceil((kept + new_tokens) / BLOCK_SIZE).astype(np.int64)
    assert handed.size == int((target - depth).sum())
