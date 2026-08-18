# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Where a live cache position's score comes from (budget regime).

Under a fixed KV budget nothing is locked in, so at every eviction EVERY live
position competes — including positions written many chunks ago. A score must
therefore be available for a position long after the chunk that produced it,
and the eviction methods answer that in fundamentally different ways:

* **Recompute** — the score is a function of what is already in the cache, so it
  can simply be computed again. KeyDiff is of this kind: a key's score is its
  similarity to the mean direction of the keys currently cached (paper Eq. 8), so
  nothing needs to be remembered and the score is always the method's own rather
  than a stale approximation of it. The cost is one extra read of the cached keys
  per eviction.
* **Accumulate** — the score depends on the history of queries and cannot be
  recovered from the cache. H2O is of this kind: a position's score is the
  attention mass it has received from every query so far, so the running sum must
  be carried in a per-position buffer and updated each step.
* **Persist** — keep whatever score the position's own chunk produced. This is
  what a chunk-local scorer (SnapKV, TOVA, the FastKVZip gate) allows without
  extra machinery, and it is an approximation: those scores were computed against
  per-chunk reference quantities, so ranking positions from different chunks
  together has no common scale.

All three write into the same slot-addressed buffer
(``[num_layers, num_kv_heads, slot_capacity]``, indexed by cache slot), which
the eviction writeback compacts alongside the KV. Selecting between them is
therefore a property of the SCORER, not another user-facing knob:
``make_slot_score_source`` picks recompute when the scorer can rescore the cache
and persist otherwise.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from vllm.logger import init_logger
from vllm.v1.attention.backends.ragged_layout import cluster_pages_token_major

logger = init_logger(__name__)


@dataclass(frozen=True)
class KVCacheView:
    """Read access to one request's cached keys, addressed the way the
    compressor thinks: by COMPRESSED layer index (the full-attention layers)
    and head group.

    Built per compression step by the model runner, which owns the KV tensors
    and the block table; passed down so the score source can read the cache
    without the compressor holding engine state.
    """
    #: KV cache tensors indexed by PHYSICAL layer.
    layer_kv_caches: list[torch.Tensor]
    #: ``[max_num_reqs, num_layers * num_groups, max_blocks_per_row]``.
    block_table_gpu: torch.Tensor
    #: This request's block-table row.
    row_idx: int
    #: Compressed position -> physical layer.
    compressed_layer_ids: np.ndarray
    num_groups: int
    block_size: int

    def cluster_keys(
        self,
        compressed_layer_idx: int,
        group_idx: int,
        num_positions: int,
    ) -> torch.Tensor:
        """Post-RoPE keys of one head group's live slots.

        Returns ``[page_group_size, num_positions, head_size]``, one row per KV
        head (cluster column), in slot order. The cluster's pages are strided
        across the pool, so materialising them is a genuine read; it is done at
        block granularity (the trailing partial block is read and then sliced
        off) which keeps the transient shapes to a few sizes the caching
        allocator can reuse.
        """
        layer_idx = int(self.compressed_layer_ids[compressed_layer_idx])
        kv_cache = self.layer_kv_caches[layer_idx]
        block_size = self.block_size
        num_blocks = (num_positions + block_size - 1) // block_size
        block_ids = self.block_table_gpu[
            self.row_idx,
            layer_idx * self.num_groups + group_idx,
            :num_blocks,
        ].long()
        # [n_blocks, block_size, page_group_size, head_size]
        pages = cluster_pages_token_major(kv_cache, block_ids, key_only=True)
        n_blocks, blk, page_group_size, head_size = pages.shape
        keys = pages.permute(2, 0, 1, 3).reshape(
            page_group_size, n_blocks * blk, head_size)
        return keys[:, :num_positions]


@dataclass
class ChunkScoreInputs:
    """Everything a score source may read for one chunk."""
    #: ``[num_layers, num_kv_heads, chunk_len]`` scores the scorer produced for
    #: the tokens just written. Zero-width when the source does not need them.
    pending: torch.Tensor
    #: ``[num_layers, num_groups]`` pre-chunk kept length per (layer, group).
    prev_lens_cpu: np.ndarray
    prev_lens_device: torch.Tensor
    chunk_len: int
    #: Read access to the cached keys; required by sources that rescore.
    cache_view: KVCacheView | None


@dataclass(frozen=True)
class SlotFillTarget:
    """The slot-addressed buffer a source writes, plus the maps to address it."""
    #: ``[num_layers, num_kv_heads, slot_capacity]``.
    buffer: torch.Tensor
    #: The same storage as ``[num_layers * num_kv_heads, slot_capacity]``, for
    #: addressing by member row.
    flat: torch.Tensor
    #: ``[num_layers * num_kv_heads]`` member row -> flat cluster id.
    member_to_cluster: torch.Tensor
    #: ``[num_clusters, page_group_size]`` cluster + column -> member row, on the
    #: CPU so the per-cluster loop does not synchronise on every iteration.
    #: ``-1`` marks a column no member occupies.
    cluster_members_cpu: np.ndarray
    num_layers: int
    num_kv_heads: int
    num_groups: int
    neg_inf: float


class SlotScoreSource(ABC):
    """Fills the slot-addressed score buffer for one chunk."""

    #: Stable identifier for logging.
    name: str
    #: Whether the per-chunk scorer must run at all. A source that recomputes
    #: from the cache does not read the chunk's own scores, so the scorer's
    #: forward pass and its buffer write are skipped entirely.
    needs_chunk_scores: bool

    @abstractmethod
    def describe(self) -> str:
        """One sentence naming what this source computes, for the startup log.
        The choice changes what the eviction actually ranks, so it is reported
        rather than left implicit — but only by the caller that knows the regime
        consumes it (the ratio regime never does)."""

    @abstractmethod
    def fill(self, target: SlotFillTarget, inputs: ChunkScoreInputs) -> None:
        """Bring ``target.buffer`` up to date for every live slot.

        A source must leave a valid score at every slot in
        ``[0, prev_len + chunk_len)`` of each (layer, group), and must not read
        or leave meaning in slots beyond it: the eviction writeback blanks the
        tail, and the keep decision masks it.
        """


class PersistedChunkScores(SlotScoreSource):
    """Keep the score each position's own chunk produced.

    The fresh chunk's scores are scattered to ``[prev_len, prev_len + chunk)``
    and earlier slots are left as they are. Cheap — no extra reads — but the
    ranking mixes scores calibrated against different chunks, so it is an
    approximation for any scorer whose score is relative to its chunk.
    """

    name = "persist"
    needs_chunk_scores = True

    def describe(self) -> str:
        return ("keeping each position's own chunk score (the scorer cannot "
                "rescore cached positions); scores from different chunks are "
                "ranked together, which is an approximation")

    def fill(self, target: SlotFillTarget, inputs: ChunkScoreInputs) -> None:
        pending = inputs.pending
        chunk_len = pending.shape[-1]
        # Members of one cluster share its length, so a member's write offset is
        # a lookup of its cluster's pre-chunk length.
        member_offset = inputs.prev_lens_device.reshape(-1)[
            target.member_to_cluster].view(
                target.num_layers, target.num_kv_heads, 1)
        write_idx = member_offset + torch.arange(
            chunk_len, device=pending.device, dtype=member_offset.dtype
        ).view(1, 1, chunk_len)
        target.buffer.scatter_(2, write_idx, pending)


class RecomputedCacheScores(SlotScoreSource):
    """Recompute every live position's score from the cached keys.

    This is what a key-similarity method (KeyDiff) actually specifies: the score
    is relative to the keys currently in the cache, so it changes as the cache
    changes and is only correct when computed against the present cache. Because
    the keys are already stored, nothing else has to be — the chunk's own scores
    are not even needed, so the per-chunk scorer is skipped.

    The cost is one read of the request's cached keys per eviction, ordered
    (layer, group) so only one group's keys are materialised at a time. The
    eviction writeback that follows already gathers and rewrites both keys and
    values, so this adds a fraction of traffic that is already being paid.
    """

    name = "recompute"
    needs_chunk_scores = False

    def __init__(self, scorer: nn.Module) -> None:
        self.scorer = scorer

    def describe(self) -> str:
        return (
            f"rescoring every live position from the cached keys with "
            f"'{getattr(self.scorer, 'name', type(self.scorer).__name__)}' "
            "at each eviction (its score is relative to the cache, so it is "
            "recomputed rather than stored)")

    def fill(self, target: SlotFillTarget, inputs: ChunkScoreInputs) -> None:
        view = inputs.cache_view
        if view is None:
            raise RuntimeError(
                "RecomputedCacheScores needs a KVCacheView: the score is a "
                "function of the cached keys, so the keep decision cannot run "
                "without read access to them.")
        live_lens = inputs.prev_lens_cpu + inputs.chunk_len
        flat = target.flat
        neg_inf = target.neg_inf
        for static_idx in range(target.num_layers):
            for group_idx in range(target.num_groups):
                cluster_id = static_idx * target.num_groups + group_idx
                rows = target.cluster_members_cpu[cluster_id]
                if (rows < 0).any():
                    continue  # Empty cluster: no member holds these slots.
                num_positions = int(live_lens[static_idx, group_idx])
                if num_positions == 0:
                    flat[rows] = neg_inf
                    continue
                keys = view.cluster_keys(
                    static_idx, group_idx, num_positions)
                scores = self.scorer.score_cached_keys(keys)
                flat[rows, :num_positions] = scores.to(flat.dtype)
                # Beyond the live extent nothing is cached; keep it unselectable
                # so a stale score can never win a rank.
                if num_positions < flat.shape[-1]:
                    flat[rows, num_positions:] = neg_inf


#: Registry, so a new source is one subclass plus one entry.
_SOURCES: dict[str, type[SlotScoreSource]] = {
    PersistedChunkScores.name: PersistedChunkScores,
    RecomputedCacheScores.name: RecomputedCacheScores,
}

#: Valid source names, for logging and tests.
SLOT_SCORE_SOURCES: tuple[str, ...] = tuple(_SOURCES)


def make_slot_score_source(scorer: nn.Module | None) -> SlotScoreSource:
    """Pick the score source the scorer supports.

    Not a user-facing choice: a scorer either can rescore cached positions or it
    cannot, and picking the wrong one is either impossible (recompute without
    ``score_cached_keys``) or a silent accuracy loss (persist when the method
    specifies a cache-relative score). The caller logs ``describe()`` if the
    active regime consumes the source at all.
    """
    if scorer is not None and getattr(scorer, "rescores_cache", False):
        return RecomputedCacheScores(scorer)
    return PersistedChunkScores()
