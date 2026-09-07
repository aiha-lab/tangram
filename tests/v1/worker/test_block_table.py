# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behaviour tests for BlockTable and its ragged-paging variant.

The block table decides which physical KV block each (request, head group,
slot) holds, so a wrong entry here does not crash -- it reads someone else's
KV. It had no tests, which made every edit to it unverifiable; these pin the
observable behaviour of both layouts so a refactor of the class has something
to be checked against.

Everything runs on CPU with no model. Run without the root conftest, which
imports a package this fork does not install:

    python -m pytest --noconftest -q tests/v1/worker/test_block_table.py
"""
import numpy as np
import pytest
import torch

from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable
from vllm.v1.worker.ragged_block_table import RaggedBlockTable

BLOCK_SIZE = 4
MAX_NUM_REQS = 3
MAX_BLOCKS_PER_REQ = 6
MAX_BATCHED_TOKENS = 32
NUM_HEAD_GROUPS = 4
GROUPS_PER_LAYER = 2
NUM_LAYERS = NUM_HEAD_GROUPS // GROUPS_PER_LAYER


BASE_KWARGS = dict(
    block_size=BLOCK_SIZE,
    max_num_reqs=MAX_NUM_REQS,
    max_num_blocks_per_req=MAX_BLOCKS_PER_REQ,
    max_num_batched_tokens=MAX_BATCHED_TOKENS,
    pin_memory=False,
    device=torch.device("cpu"),
    kernel_block_size=BLOCK_SIZE,
    cp_kv_cache_interleave_size=1,
)


def make_table(**overrides) -> BlockTable:
    return BlockTable(**{**BASE_KWARGS, **overrides})


def make_ragged(**overrides) -> RaggedBlockTable:
    kwargs = dict(
        num_head_groups=NUM_HEAD_GROUPS,
        num_head_groups_per_layer=GROUPS_PER_LAYER,
    )
    kwargs.update(overrides)
    return RaggedBlockTable(**{**BASE_KWARGS, **kwargs})


# --- Shape: the group axis is what ragged paging adds ----------------------


def test_plain_table_is_two_dimensional():
    table = make_table()

    assert table.ragged is False
    assert table.block_table.np.shape == (MAX_NUM_REQS, MAX_BLOCKS_PER_REQ)
    assert table.num_blocks_per_row.shape == (MAX_NUM_REQS,)
    assert table.slot_mapping.np.shape == (MAX_BATCHED_TOKENS,)


def test_ragged_table_carries_a_group_axis():
    table = make_ragged()

    assert table.ragged is True
    assert table.block_table.np.shape == (
        MAX_NUM_REQS,
        NUM_HEAD_GROUPS,
        MAX_BLOCKS_PER_REQ,
    )
    assert table.num_blocks_per_row.shape == (MAX_NUM_REQS, NUM_HEAD_GROUPS)
    assert table.slot_mapping.np.shape == (NUM_HEAD_GROUPS, MAX_BATCHED_TOKENS)


def test_ragged_defaults_groups_per_layer_to_the_total():
    """Single-layer unit tests may omit it; production always passes it."""
    table = make_ragged(num_head_groups_per_layer=None)

    assert table.num_head_groups_per_layer == NUM_HEAD_GROUPS


def test_ragged_rejects_a_kernel_block_size_of_its_own():
    with pytest.raises(AssertionError, match="kernel_block_size"):
        make_ragged(kernel_block_size=BLOCK_SIZE // 2)


def test_ragged_rejects_groups_that_do_not_tile_the_layers():
    with pytest.raises(AssertionError, match="must divide"):
        make_ragged(num_head_groups=6, num_head_groups_per_layer=4)


# --- append_row: ids are laid out group-major to a uniform depth ----------


def test_plain_append_row_fills_one_row_left_to_right():
    table = make_table()

    table.append_row([10, 11], row_idx=1)
    table.append_row([12], row_idx=1)

    assert table.num_blocks_per_row[1] == 3
    np.testing.assert_array_equal(table.block_table.np[1, :3], [10, 11, 12])


def test_ragged_append_row_gives_every_group_the_same_depth():
    """The caller hands over ``num_head_groups x num_new`` ids in group-major
    order, and every group ends at the same fill: uniform depth is what makes
    the flat block-id list reconstructible."""
    table = make_ragged()

    table.append_row([100, 101, 102, 103], row_idx=0)

    np.testing.assert_array_equal(table.num_blocks_per_row[0], [1, 1, 1, 1])
    np.testing.assert_array_equal(table.block_table.np[0, :, 0], [100, 101, 102, 103])


def test_ragged_append_row_tops_up_uneven_groups_first():
    """After compression the groups sit at different depths, and an append
    levels them: group 0 takes its shortfall first, then group 1, and so on."""
    table = make_ragged()
    table.append_row([1, 2, 3, 4], row_idx=0)
    table.num_blocks_per_row[0] = [2, 1, 1, 1]
    table.block_table.np[0, 0, 1] = 9

    # Target depth 2 for all four groups: group 0 needs none, the rest one each.
    table.append_row([20, 21, 22], row_idx=0)

    np.testing.assert_array_equal(table.num_blocks_per_row[0], [2, 2, 2, 2])
    np.testing.assert_array_equal(table.block_table.np[0, :, 1], [9, 20, 21, 22])


def test_ragged_append_row_rejects_a_count_that_cannot_level_the_groups():
    """A total that leaves no integer depth means the allocator and the append
    disagree, which would corrupt the (row, group, slot) -> id mapping."""
    table = make_ragged()

    with pytest.raises(RuntimeError, match="not divisible"):
        table.append_row([1, 2, 3], row_idx=0)


def test_ragged_append_row_refuses_to_shrink():
    table = make_ragged()
    table.num_blocks_per_row[0] = [3, 0, 0, 0]

    with pytest.raises(RuntimeError, match="cannot shrink"):
        table.append_row([1], row_idx=0)


def test_append_row_ignores_an_empty_list():
    table = make_ragged()

    table.append_row([], row_idx=0)

    np.testing.assert_array_equal(table.num_blocks_per_row[0], [0, 0, 0, 0])


def test_add_row_resets_the_row_first():
    table = make_ragged()
    table.append_row([1, 2, 3, 4], row_idx=2)

    table.add_row([5, 6, 7, 8], row_idx=2)

    np.testing.assert_array_equal(table.num_blocks_per_row[2], [1, 1, 1, 1])
    np.testing.assert_array_equal(table.block_table.np[2, :, 0], [5, 6, 7, 8])


# --- move / swap: whole rows, group axis included -------------------------


def test_plain_move_row_copies_the_filled_prefix():
    table = make_table()
    table.append_row([7, 8], row_idx=0)

    table.move_row(src=0, tgt=2)

    assert table.num_blocks_per_row[2] == 2
    np.testing.assert_array_equal(table.block_table.np[2, :2], [7, 8])


def test_ragged_move_row_copies_every_group():
    table = make_ragged()
    table.append_row([1, 2, 3, 4], row_idx=0)
    table.num_blocks_per_row[0] = [1, 1, 0, 1]

    table.move_row(src=0, tgt=1)

    np.testing.assert_array_equal(table.num_blocks_per_row[1], [1, 1, 0, 1])
    np.testing.assert_array_equal(table.block_table.np[1, :, 0], [1, 2, 3, 4])


def test_ragged_swap_row_exchanges_ids_and_counts():
    table = make_ragged()
    table.append_row([1, 2, 3, 4], row_idx=0)
    table.append_row([5, 6, 7, 8], row_idx=1)
    table.num_blocks_per_row[1] = [1, 1, 1, 0]

    table.swap_row(0, 1)

    np.testing.assert_array_equal(table.block_table.np[0, :, 0], [5, 6, 7, 8])
    np.testing.assert_array_equal(table.block_table.np[1, :, 0], [1, 2, 3, 4])
    np.testing.assert_array_equal(table.num_blocks_per_row[0], [1, 1, 1, 0])
    np.testing.assert_array_equal(table.num_blocks_per_row[1], [1, 1, 1, 1])


# --- snapshot / restore: the exact layout, not a re-derived one -----------


def test_snapshot_and_restore_return_the_non_uniform_layout():
    """After compression a row's per-group depths differ, and add_row cannot
    rebuild that: it fills uniformly. A dropped request's row is captured
    verbatim and written back on re-add, so the KV stays where it lives."""
    table = make_ragged()
    table.append_row([1, 2, 3, 4], row_idx=0)
    table.num_blocks_per_row[0] = [1, 0, 1, 0]
    snapshot = table.snapshot_row(0)

    # Six ids level the four groups (depths 1, 0, 1, 0) back to a depth of 2.
    table.append_row([9, 9, 9, 9, 9, 9], row_idx=0)
    table.restore_row(0, snapshot)

    np.testing.assert_array_equal(table.num_blocks_per_row[0], [1, 0, 1, 0])
    np.testing.assert_array_equal(table.block_table.np[0, :, 0], [1, 2, 3, 4])


def test_snapshot_row_copies_rather_than_aliases():
    table = make_ragged()
    table.append_row([1, 2, 3, 4], row_idx=0)
    block_ids, counts = table.snapshot_row(0)

    table.block_table.np[0, 0, 0] = 99
    table.num_blocks_per_row[0, 0] = 5

    assert block_ids[0, 0] == 1
    assert counts[0] == 1


def test_snapshot_row_exists_only_on_the_ragged_table():
    """The plain layout has one depth per request, so there is nothing a
    snapshot could preserve that add_row cannot rebuild -- the method is absent
    rather than guarded."""
    assert not hasattr(make_table(), "snapshot_row")


# --- slot mapping ---------------------------------------------------------


def test_plain_slot_mapping_maps_positions_through_the_block_table():
    table = make_table()
    table.append_row([5, 6], row_idx=0)

    table.compute_slot_mapping(
        req_indices=np.array([0, 0, 0]), positions=np.array([0, 3, 4])
    )

    # position 0 and 3 are in block 5; position 4 starts block 6.
    np.testing.assert_array_equal(
        table.slot_mapping.np[:3], [5 * BLOCK_SIZE, 5 * BLOCK_SIZE + 3, 6 * BLOCK_SIZE]
    )


def test_ragged_slot_mapping_broadcasts_shared_positions_to_every_group():
    """Before compression every group holds the same token positions, so a 1D
    positions array applies to all of them -- against each group's own blocks."""
    table = make_ragged()
    table.append_row([10, 20, 30, 40], row_idx=0)

    table.compute_slot_mapping(
        req_indices=np.array([0, 0]), positions=np.array([0, 1])
    )

    expected = np.array(
        [[10 * BLOCK_SIZE, 10 * BLOCK_SIZE + 1],
         [20 * BLOCK_SIZE, 20 * BLOCK_SIZE + 1],
         [30 * BLOCK_SIZE, 30 * BLOCK_SIZE + 1],
         [40 * BLOCK_SIZE, 40 * BLOCK_SIZE + 1]]
    )
    np.testing.assert_array_equal(table.slot_mapping.np[:, :2], expected)


def test_ragged_slot_mapping_takes_per_group_positions():
    """After compression each group is at its own length, so the write position
    differs per group and arrives as a 2D array."""
    table = make_ragged()
    table.append_row([10, 20, 30, 40], row_idx=0)

    positions = np.array([[0], [1], [2], [3]])
    table.compute_slot_mapping(req_indices=np.array([0]), positions=positions)

    np.testing.assert_array_equal(
        table.slot_mapping.np[:, 0],
        [10 * BLOCK_SIZE, 20 * BLOCK_SIZE + 1, 30 * BLOCK_SIZE + 2, 40 * BLOCK_SIZE + 3],
    )


def test_ragged_slot_mapping_rejects_a_wrong_group_count():
    table = make_ragged()

    with pytest.raises(AssertionError, match="num_head_groups"):
        table.compute_slot_mapping(
            req_indices=np.array([0]), positions=np.zeros((2, 1), dtype=np.int64)
        )


# --- commit: the ragged buffers are group-major on both axes --------------


def test_ragged_commit_copies_the_group_axis_to_the_device():
    table = make_ragged()
    table.append_row([1, 2, 3, 4], row_idx=0)
    table.slot_mapping.np[:, :2] = 7

    table.commit_block_table(num_reqs=1)
    table.commit_slot_mapping(num_tokens=2)

    np.testing.assert_array_equal(
        table.block_table.gpu[0, :, 0].numpy(), [1, 2, 3, 4]
    )
    assert (table.slot_mapping.gpu[:, :2] == 7).all()


def test_plain_commit_copies_the_flat_buffers():
    table = make_table()
    table.append_row([1, 2], row_idx=0)
    table.slot_mapping.np[:2] = 7

    table.commit_block_table(num_reqs=1)
    table.commit_slot_mapping(num_tokens=2)

    np.testing.assert_array_equal(table.block_table.gpu[0, :2].numpy(), [1, 2])
    assert (table.slot_mapping.gpu[:2] == 7).all()


# --- compaction after compression: keep the front, free the tail ----------


def test_compact_frees_the_trailing_blocks_of_every_layer_group():
    table = make_ragged()
    table.append_row(list(range(1, 1 + NUM_HEAD_GROUPS * 3)), row_idx=0)

    # Layer 0 keeps 3 and 1 blocks, layer 1 keeps 2 and 3.
    freed = table.compact_after_compress_all_layers(
        row_idx=0,
        num_head_groups_per_layer=GROUPS_PER_LAYER,
        new_num_blocks_per_layer=np.array([[3, 1], [2, 3]]),
    )

    np.testing.assert_array_equal(table.num_blocks_per_row[0], [3, 1, 2, 3])
    # append_row lays ids out group-major, so group 1 holds [4, 5, 6] and group
    # 2 holds [7, 8, 9]: group 1 gives up its last two, group 2 its last one,
    # and groups 0 and 3 keep everything.
    assert sorted(freed.tolist()) == [5, 6, 9]
    assert (table.block_table.np[0, 1, 1:] == 0).all()


def test_compact_cannot_grow_a_group():
    table = make_ragged()
    table.append_row([1, 2, 3, 4], row_idx=0)

    with pytest.raises(RuntimeError, match="cannot grow"):
        table.compact_after_compress_all_layers(
            row_idx=0,
            num_head_groups_per_layer=GROUPS_PER_LAYER,
            new_num_blocks_per_layer=np.array([[2, 1], [1, 1]]),
        )


def test_compact_rejects_a_shape_that_is_not_layers_by_groups():
    table = make_ragged()

    with pytest.raises(ValueError, match="num_layers"):
        table.compact_after_compress_all_layers(
            row_idx=0,
            num_head_groups_per_layer=GROUPS_PER_LAYER,
            new_num_blocks_per_layer=np.array([1, 1, 1, 1]),
        )


def test_compact_exists_only_on_the_ragged_table():
    assert not hasattr(make_table(), "compact_after_compress_all_layers")


# --- sliding window: keep the tail, null the front -----------------------


def test_null_front_frees_leading_blocks_and_keeps_the_counts():
    """The surviving tail must not move: nulling the front rather than
    compacting it is what keeps every position -> block mapping intact, so the
    output is unchanged."""
    table = make_ragged()
    table.append_row(list(range(1, 1 + NUM_HEAD_GROUPS * 3)), row_idx=0)
    before = table.num_blocks_per_row[0].copy()

    freed = table.null_front_blocks_sliding(
        row_idx=0,
        sliding_layer_ids=np.array([1]),
        num_head_groups_per_layer=GROUPS_PER_LAYER,
        num_skipped_blocks=2,
    )

    # Layer 1 owns flat groups 2 and 3, holding [7, 8, 9] and [10, 11, 12];
    # each gives up its first two blocks.
    assert sorted(freed.tolist()) == [7, 8, 10, 11]
    np.testing.assert_array_equal(table.num_blocks_per_row[0], before)
    assert (table.block_table.np[0, 2:4, :2] == 0).all()
    # Layer 0 is a full-attention layer and is untouched.
    assert (table.block_table.np[0, :2, 0] != 0).all()


def test_null_front_does_nothing_without_skipped_blocks():
    table = make_ragged()
    table.append_row([1, 2, 3, 4], row_idx=0)

    freed = table.null_front_blocks_sliding(
        row_idx=0,
        sliding_layer_ids=np.array([1]),
        num_head_groups_per_layer=GROUPS_PER_LAYER,
        num_skipped_blocks=0,
    )

    assert freed.size == 0
    np.testing.assert_array_equal(table.block_table.np[0, :, 0], [1, 2, 3, 4])


def test_null_front_clamps_to_what_a_group_actually_holds():
    """Asking to skip more blocks than a group has must not reach into columns
    that were never allocated."""
    table = make_ragged()
    table.append_row([1, 2, 3, 4], row_idx=0)

    freed = table.null_front_blocks_sliding(
        row_idx=0,
        sliding_layer_ids=np.array([1]),
        num_head_groups_per_layer=GROUPS_PER_LAYER,
        num_skipped_blocks=MAX_BLOCKS_PER_REQ + 5,
    )

    assert sorted(freed.tolist()) == [3, 4]


def test_null_front_exists_only_on_the_ragged_table():
    assert not hasattr(make_table(), "null_front_blocks_sliding")


# --- clear ---------------------------------------------------------------


def test_clear_zeroes_ids_and_counts():
    table = make_ragged()
    table.append_row([1, 2, 3, 4], row_idx=0)

    table.clear()

    assert (table.block_table.np == 0).all()
    assert (table.num_blocks_per_row == 0).all()


# --- MultiGroupBlockTable ------------------------------------------------


def make_multi(**overrides) -> MultiGroupBlockTable:
    kwargs = dict(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=BLOCK_SIZE * MAX_BLOCKS_PER_REQ,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    kwargs.update(overrides)
    return MultiGroupBlockTable(**kwargs)


def test_multi_group_ragged_collapses_to_a_single_table():
    """Ragged paging absorbs the layer axis into the flat group index, so there
    is one 3D table rather than one per KV cache group."""
    multi = make_multi(
        num_head_groups=NUM_HEAD_GROUPS,
        num_head_groups_per_layer=GROUPS_PER_LAYER,
    )

    assert len(multi.block_tables) == 1
    assert multi[0].ragged is True


def test_multi_group_plain_keeps_one_table_per_kv_cache_group():
    multi = make_multi(block_sizes=[BLOCK_SIZE, BLOCK_SIZE],
                       kernel_block_sizes=[BLOCK_SIZE, BLOCK_SIZE])

    assert len(multi.block_tables) == 2
    assert multi[0].ragged is False


def test_multi_group_ragged_rejects_more_than_one_kv_cache_group():
    with pytest.raises(AssertionError, match="single KVCacheGroupSpec"):
        make_multi(
            block_sizes=[BLOCK_SIZE, BLOCK_SIZE],
            kernel_block_sizes=[BLOCK_SIZE, BLOCK_SIZE],
            num_head_groups=NUM_HEAD_GROUPS,
        )


def test_multi_group_ragged_routes_rows_to_the_single_table():
    multi = make_multi(
        num_head_groups=NUM_HEAD_GROUPS,
        num_head_groups_per_layer=GROUPS_PER_LAYER,
    )

    multi.append_row(([1, 2, 3, 4],), row_idx=0)

    np.testing.assert_array_equal(multi[0].block_table.np[0, :, 0], [1, 2, 3, 4])


def test_multi_group_plain_fans_rows_out_per_group():
    multi = make_multi(block_sizes=[BLOCK_SIZE, BLOCK_SIZE],
                       kernel_block_sizes=[BLOCK_SIZE, BLOCK_SIZE])

    multi.append_row(([1], [2]), row_idx=0)

    assert multi[0].block_table.np[0, 0] == 1
    assert multi[1].block_table.np[0, 0] == 2


def test_multi_group_snapshot_and_restore_round_trip():
    multi = make_multi(
        num_head_groups=NUM_HEAD_GROUPS,
        num_head_groups_per_layer=GROUPS_PER_LAYER,
    )
    multi.append_row(([1, 2, 3, 4],), row_idx=0)
    multi[0].num_blocks_per_row[0] = [1, 0, 1, 0]
    snapshot = multi.snapshot_row(0)

    multi.add_row(([9, 9, 9, 9],), row_idx=0)
    multi.restore_row(0, snapshot)

    np.testing.assert_array_equal(multi[0].num_blocks_per_row[0], [1, 0, 1, 0])
    np.testing.assert_array_equal(multi[0].block_table.np[0, :, 0], [1, 2, 3, 4])


def test_multi_group_snapshot_is_ragged_only():
    """MultiGroupBlockTable delegates, so a plain table's missing method is
    what rejects the call."""
    with pytest.raises(AttributeError, match="snapshot_row"):
        make_multi().snapshot_row(0)
