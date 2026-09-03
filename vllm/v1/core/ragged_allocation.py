# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How many blocks a ragged request's head groups need.

A block holds one head group, and compression leaves every group its own
length, so a request occupies the SUM over groups::

    blocks = sum_g ceil(len_g / block_size)

Pricing it as a rectangle over the longest group instead -- ``ceil(max_g len_g
/ block_size) x num_groups`` -- re-takes at the next allocation exactly the
pages compression just freed, because the shortest group is charged for the
longest one.

The scheduler sizes allocations with this and the worker fills the block table
to it. Keeping the rule in one module is what makes the two agree: a request
is filled by the same arithmetic that priced it.
"""
from __future__ import annotations

import numpy as np

from vllm.utils.math_utils import cdiv


def group_block_depths(
    lengths: int | np.ndarray,
    block_size: int,
    num_groups: int,
) -> np.ndarray:
    """Blocks each head group needs in order to hold ``lengths`` tokens.

    ``lengths`` is either one length shared by every group -- what a request
    holds before compression, where the sum and the rectangle agree -- or one
    length per group, which is what compression leaves behind.
    """
    if not isinstance(lengths, np.ndarray):
        return np.full(num_groups, cdiv(int(lengths), block_size),
                       dtype=np.int64)

    depths = lengths.astype(np.int64, copy=False)
    if depths.size != num_groups:
        raise ValueError(
            f"per-group lengths have {depths.size} entries but "
            f"num_head_groups is {num_groups}.")
    # Ceiling division on integers; no float round trip to misround a length
    # that is already a block multiple.
    return -(-depths // block_size)


def blocks_for(
    lengths: int | np.ndarray,
    block_size: int,
    num_groups: int,
) -> int:
    """Total blocks a ragged request occupies -- the sum over its groups."""
    return int(group_block_depths(lengths, block_size, num_groups).sum())
