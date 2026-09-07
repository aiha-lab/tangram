# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How many slots each (layer, head group) keeps after an eviction.

The single source of truth for that number, and pure arithmetic over the arrays
handed in -- no KV cache, no request state. The compressor caches the result and
the executor reads that cache, so the two agree by construction.

Four rules shape every count, and each is correctness, not preference. A kept
length is block-aligned, a page being the unit of reclamation. The floor cannot
exceed what the cache holds nor the budget. A budget ceiling rounds DOWN where
the selection rounds UP, so a page is never cut in half. And a pooling scope
shares one total over its span, which is why block quantisation is settled per
span rather than per entry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KeepDecision:
    """Per-chunk geometry the executor consumes, in cache-slot coordinates.

    An entry's eval region is the ``eval_len`` slots at ``sink_size + locked``;
    sink, locked prefix and trailing ``tail_size`` are kept regardless of score.
    The kept count and positions live in the per-entry caches, not here: the
    scope's threshold is a ``BudgetScope`` internal.
    """
    sink_size: int
    #: Trailing always-kept slots: the recent window (ratio) or the whole
    #: fresh chunk (budget, unless it is configured to be evictable).
    tail_size: int
    adjusted_ratio: float
    eval_len: int = 0
    #: Per-entry hard cap on the kept length, ``None`` under the ratio regime.
    budget_tokens: int | None = None


def _apportion_blocks(
    want: np.ndarray,
    base: np.ndarray,
    remainder: np.ndarray,
    total: int,
    block_size: int,
) -> np.ndarray:
    """Share one pooled ``total`` over a span's (layer, group) entries.

    Largest-remainder apportionment: each entry takes the block FLOOR of its
    demand (``base``, from ``want``), then leftover blocks go to whoever
    flooring shortchanged most (``remainder``, which is the hand-out order).
    Flooring rather than rounding up is what makes ``sum <= total``
    structural -- rounding up needs the same amount taken back, with no rule
    for whom to take it from. Ties break on the lower index, so a rerun decides
    the same way.
    """
    k = np.minimum(base, want).astype(np.int64)

    # ``floor_min`` upstream can push the floors past the total. A floor is a
    # request and the budget a limit, so shave blocks off the largest holder.
    while int(k.sum()) > total:
        biggest = int(np.argmax(k))
        if k[biggest] <= 0:
            break
        k[biggest] = max(0, int(k[biggest]) - block_size)

    # Leftover blocks go most-shortchanged entry first, while any can take one.
    order = np.lexsort((np.arange(len(k)), -remainder))
    spare = (total - int(k.sum())) // block_size
    handed_out = True
    while spare > 0 and handed_out:
        handed_out = False
        for entry in order:
            if spare <= 0:
                break
            if k[entry] + block_size <= want[entry]:
                k[entry] += block_size
                spare -= 1
                handed_out = True

    # A mid-block ``want`` is unreachable above; pay it in the same order.
    slack = total - int(k.sum())
    for entry in order:
        if slack <= 0:
            break
        owed = min(int(want[entry]) - int(k[entry]), slack)
        if owed > 0:
            k[entry] += owed
            slack -= owed
    return k


def kept_lengths_from_demand(
    *,
    decision: KeepDecision,
    total_seen: np.ndarray,
    locked_counts: np.ndarray,
    k_new_counts: np.ndarray | None,
    real_eval_lens: np.ndarray,
    block_size: int,
    floor_min: int,
    pooled_span: str | None,
    per_group_capacity: int,
) -> np.ndarray:
    """Per-(layer, group) post-evict kept lengths, ``[num_layers, num_groups]``.

    Three passes, because a pooling scope shares one total over a span: collect
    each entry's block-rounded demand, enforce the budget (per span when the
    scope pools, else per entry), turn the counts back into lengths.

    ``pooled_span`` is ``None`` under the ratio regime and whenever the scope
    does not pool. ``per_group_capacity`` is the workspace's physical ceiling --
    the budget itself under ``uniform``, larger when pooling.
    """
    num_layers, num_groups = total_seen.shape
    sink_size = decision.sink_size
    tail_size = decision.tail_size
    budget_tokens = decision.budget_tokens


    # Pass 1 — per entry: block-rounded demand, floor, dropped remainder.
    want = np.zeros((num_layers, num_groups), dtype=np.int64)
    base = np.zeros((num_layers, num_groups), dtype=np.int64)
    remainder = np.zeros((num_layers, num_groups), dtype=np.int64)
    for layer_idx in range(num_layers):
        for group_idx in range(num_groups):
            total_seen_g = int(total_seen[layer_idx, group_idx])
            locked_count = int(locked_counts[layer_idx, group_idx])
            eval_len_g = int(real_eval_lens[layer_idx, group_idx])
            if eval_len_g <= 0:
                continue
            # adjusted_ratio == 0 ⇒ no sort cached, keep none.
            k_new = (int(k_new_counts[layer_idx, group_idx])
                     if k_new_counts is not None else 0)
            kept_now = (
                sink_size + locked_count + k_new + tail_size)
            # A floor cannot exceed what the cache holds, nor the budget.
            target_floor = min(floor_min, total_seen_g)
            if budget_tokens is not None:
                target_floor = min(target_floor, budget_tokens)
            if kept_now < target_floor:
                extra = min(
                    target_floor - kept_now,
                    eval_len_g - k_new)
                if extra > 0:
                    k_new += extra
            k_aligned = (
                ((k_new + block_size - 1) // block_size)
                * block_size)
            k_aligned = min(k_aligned, eval_len_g)
            if budget_tokens is not None:
                # The selection rounds UP for page contiguity, so the
                # ceiling rounds DOWN rather than cutting mid-block. Under
                # ``uniform`` the capacity IS the budget; under a pooling
                # scope it is the physical ceiling and pass 2 enforces the
                # budget.
                room_g = (per_group_capacity - sink_size - locked_count
                          - tail_size)
                k_aligned = min(
                    k_aligned, max(0, (room_g // block_size) * block_size))
            want[layer_idx, group_idx] = k_aligned
            base[layer_idx, group_idx] = min(
                (k_new // block_size) * block_size, k_aligned)
            remainder[layer_idx, group_idx] = (
                k_new - base[layer_idx, group_idx])

    # Pass 2 — per span: the total sums what the BUDGET, not the capacity,
    # leaves each entry, so one wanting less leaves the rest. The pooling.
    if pooled_span is None:
        keep_counts = want
    else:
        budget_room = np.maximum(
            budget_tokens - sink_size - locked_counts - tail_size, 0)
        flat_shape = num_layers * num_groups
        spans = (
            [np.arange(flat_shape)] if pooled_span == "global"
            else [np.arange(l * num_groups, (l + 1) * num_groups)
                  for l in range(num_layers)])
        flat_want = want.reshape(-1)
        flat_base = base.reshape(-1)
        flat_remainder = remainder.reshape(-1)
        flat_room = budget_room.reshape(-1)
        keep_counts = np.zeros(flat_shape, dtype=np.int64)
        for members in spans:
            keep_counts[members] = _apportion_blocks(
                flat_want[members], flat_base[members],
                flat_remainder[members],
                int(flat_room[members].sum()), block_size)
        keep_counts = keep_counts.reshape(num_layers, num_groups)

    # Pass 3 — per entry: counts back to lengths.
    kept_lengths = np.zeros(
        (num_layers, num_groups), dtype=np.int32)
    for layer_idx in range(num_layers):
        for group_idx in range(num_groups):
            new_locked = (int(locked_counts[layer_idx, group_idx])
                          + int(keep_counts[layer_idx, group_idx]))
            kept_length = sink_size + new_locked + tail_size
            total_seen_g = int(total_seen[layer_idx, group_idx])
            if kept_length > total_seen_g:
                kept_length = total_seen_g
            kept_lengths[layer_idx, group_idx] = kept_length
    return kept_lengths
