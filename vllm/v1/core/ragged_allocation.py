# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How many blocks a ragged request's head groups need.

A block holds one head group, and compression leaves every group its own
length, so a request occupies about the SUM over groups::

    blocks ~= sum_g ceil(len_g / block_size)

Pricing it as a rectangle over the longest group instead -- ``ceil(max_g len_g
/ block_size) x num_groups`` -- re-takes at the next allocation exactly the
pages compression just freed, because the shortest group is charged for the
longest one.

The groups are topped up together, at the longest group's block edge. Each
group gets room up to where the longest group's last block ends, less the
whole blocks it is shorter (block size 4)::

    longest  |####|####|##..|        3 blocks
    group g  |####|#...|             3 - (10 - 5) // 4 = 2 blocks

Every group stores every decoded token, so the gaps -- and the blocks they
spare -- stay fixed between compressions, and all groups cross an edge on the
same step: once per block of tokens, as without compression. Topping each
group up at its own edge holds the same pages within one per group, but the
edges are staggered and it allocates on almost every decode step.

The scheduler prices a request with ``blocks_for``, from two integers: the
longest group's length and ``spare_blocks``. The worker fills the block table
with ``group_block_depths``. Both are this one rule, which is what keeps the
two in agreement.
"""
from __future__ import annotations

import numpy as np

from vllm.utils.math_utils import cdiv


def spare_blocks(kept_lengths, block_size: int) -> int:
    """Whole blocks the groups are short of the longest one, summed.

    ``kept_lengths`` holds one length per head group. Zero when the groups
    agree, which is every request before compression.
    """
    kept = np.asarray(kept_lengths, dtype=np.int64)
    gaps = kept.max() - kept
    return int((gaps // block_size).sum())


def blocks_for(
    num_tokens: int, block_size: int, num_groups: int, spare: int
) -> int:
    """Total blocks a ragged request holds once its longest group needs
    ``num_tokens`` slots: the sum of ``group_block_depths``.

    ``spare`` is ``spare_blocks`` of the kept lengths; zero before compression,
    where every group holds the longest one's blocks.
    """
    return cdiv(num_tokens, block_size) * num_groups - spare


def group_block_depths(
    kept_lengths: np.ndarray,
    num_new_tokens: int,
    max_len: int,
    block_size: int,
) -> np.ndarray:
    """Blocks each head group must hold once ``num_new_tokens`` are written.

    Their sum is what the scheduler priced with ``blocks_for``.
    """
    kept = kept_lengths.astype(np.int64, copy=False)
    longest = int(kept.max())
    edge = cdiv(min(longest + num_new_tokens, max_len), block_size)
    return edge - (longest - kept) // block_size
