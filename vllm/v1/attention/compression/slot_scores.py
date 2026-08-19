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
* **Persist** — keep whatever score the position's own chunk produced. This is
  what a chunk-local scorer (SnapKV, TOVA, the FastKVZip gate) allows without
  extra machinery, and it is an approximation: those scores were computed against
  per-chunk reference quantities, so ranking positions from different chunks
  together has no common scale.

A third kind — accumulating a running score per position, as H2O does — has
no source here yet. Both existing ones write into the same slot-addressed buffer
(``[num_layers, num_kv_heads, slot_capacity]``, indexed by cache slot), which
the eviction writeback compacts alongside the KV. Selecting between them is
therefore a property of the SCORER, not a tuning knob:
``make_slot_score_source`` picks recompute when the scorer can rescore the cache
and persist otherwise. A source name can still be forced, for one purpose only —
an ablation that holds the score fixed while the retention target changes, so
the two can be measured apart (see ``make_slot_score_source``).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from vllm.logger import init_logger
from vllm.v1.attention.backends.ragged_layout import cluster_pages_token_major
from vllm.v1.attention.compression.qk_scorer_base import (
    RESCORE_INPUTS,
    RESCORE_KEYS,
    RESCORE_VALUES,
    CachedPositions,
)

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

    def materialize(
        self,
        compressed_layer_idx: int,
        group_idx: int,
        num_positions: int,
        wanted: Sequence[str],
    ) -> CachedPositions:
        """Read one head group's live slots, but only what the scorer asked for.

        ``wanted`` comes from the scorer's ``rescore_inputs``, so a key-only
        method pays for the keys alone while a method that also needs values
        gets them from the SAME read — and a future input can be added here
        without changing the scorer contract again.

        The cluster's pages are strided across the pool, so materialising them
        is a genuine read; it is done at block granularity (the trailing partial
        block is read and then sliced off), which keeps the transient shapes to
        a few sizes the caching allocator can reuse.
        """
        unknown = [name for name in wanted if name not in RESCORE_INPUTS]
        if unknown:
            raise ValueError(
                f"rescore_inputs {unknown} are not cache inputs; expected a "
                f"subset of {RESCORE_INPUTS}.")
        layer_idx = int(self.compressed_layer_ids[compressed_layer_idx])
        kv_cache = self.layer_kv_caches[layer_idx]
        block_size = self.block_size
        num_blocks = (num_positions + block_size - 1) // block_size
        block_ids = self.block_table_gpu[
            self.row_idx,
            layer_idx * self.num_groups + group_idx,
            :num_blocks,
        ].long()
        want_values = RESCORE_VALUES in wanted
        # [n_blocks, block_size, page_group_size, head_size] per KV component;
        # key_only skips the value half of the page entirely when unused.
        pages = cluster_pages_token_major(
            kv_cache, block_ids, key_only=not want_values)

        def group_major(component: torch.Tensor) -> torch.Tensor:
            n_blocks, blk, page_group_size, head_size = component.shape
            flat = component.permute(2, 0, 1, 3).reshape(
                page_group_size, n_blocks * blk, head_size)
            return flat[:, :num_positions]

        keys_pages, values_pages = (
            (pages[0], pages[1]) if want_values else (pages, None))
        return CachedPositions(
            keys=group_major(keys_pages) if RESCORE_KEYS in wanted else None,
            values=(group_major(values_pages)
                    if values_pages is not None else None),
            num_positions=num_positions,
        )


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

    def __init__(self, forced_over_recompute: bool = False) -> None:
        # True only when the scorer COULD have rescored the cache and the user
        # asked for persistence anyway (the ablation described in
        # ``make_slot_score_source``). It changes nothing this class does; it
        # only makes the startup line state why the cheaper source was taken,
        # since "the scorer cannot rescore" would be false in that case.
        self.forced_over_recompute = forced_over_recompute

    def describe(self) -> str:
        reason = ("although the scorer could rescore them — forced for an "
                  "ablation" if self.forced_over_recompute else
                  "the scorer cannot rescore cached positions")
        return (f"keeping each position's own chunk score ({reason}); scores "
                "from different chunks are ranked together, which is an "
                "approximation")

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
            f"'{self.scorer.name}' "
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
                    if (rows < 0).all():
                        continue  # Empty cluster: no member holds these slots.
                    # A member left unscored here would rank on the previous
                    # chunk's scores while its peers rank on the rescored ones.
                    raise RuntimeError(
                        f"fill: cluster {cluster_id} holds members in some "
                        f"columns but not others ({rows.tolist()}); a cluster "
                        "map must leave a cluster either full or empty.")
                num_positions = int(live_lens[static_idx, group_idx])
                if num_positions == 0:
                    flat[rows] = neg_inf
                    continue
                cached = view.materialize(
                    static_idx, group_idx, num_positions,
                    self.scorer.rescore_inputs)
                scores = self.scorer.score_cached(cached)
                flat[rows, :num_positions] = scores.to(flat.dtype)
                # Beyond the live extent nothing is cached; keep it unselectable
                # so a stale score can never win a rank.
                if num_positions < flat.shape[-1]:
                    flat[rows, num_positions:] = neg_inf


#: Valid source names, for logging and tests.
SLOT_SCORE_SOURCES: tuple[str, ...] = (
    PersistedChunkScores.name, RecomputedCacheScores.name)

#: Value asking for the source the scorer supports — the production setting.
SLOT_SCORE_SOURCE_AUTO = "auto"

#: Everything the configuration accepts: the automatic choice plus every
#: registered source, each of which can be forced for an ablation.
SLOT_SCORE_SOURCE_CHOICES: tuple[str, ...] = (
    SLOT_SCORE_SOURCE_AUTO, *SLOT_SCORE_SOURCES)


def _supports_recompute(scorer: nn.Module | None) -> bool:
    """Whether this scorer can score positions that are already in the cache."""
    return scorer is not None and getattr(scorer, "rescores_cache", False)


def make_slot_score_source(
    scorer: nn.Module | None,
    source: str = SLOT_SCORE_SOURCE_AUTO,
) -> SlotScoreSource:
    """Bind the score source, normally from what the scorer supports.

    ``source="auto"`` (the production setting) is not a user-facing choice: a
    scorer either can rescore cached positions or it cannot, and picking the
    wrong one is either impossible (recompute without ``score_cached``) or
    a silent accuracy loss (persist when the method specifies a cache-relative
    score). Hence recompute whenever the scorer offers it, persist otherwise.

    An explicit source name overrides that, which exists for ONE reason: an
    ablation. Moving a run from the ratio regime to a budget changes two things
    at once for a rescoring scorer — the retention target (and with it lock-in
    and the candidate set) and the score itself (chunk-local anchor -> whole-
    cache anchor). Forcing ``"persist"`` holds the score fixed at what the ratio
    regime also ranks, so the two effects can be measured apart. It is a
    measurement instrument, not a tuning knob: on a rescoring scorer it ranks
    positions by scores their own chunks produced, which the method does not
    specify. Forcing ``"recompute"`` on a scorer that cannot rescore is rejected
    outright rather than degraded.

    The caller logs ``describe()`` if the active regime consumes the source.
    """
    if source == SLOT_SCORE_SOURCE_AUTO:
        return (RecomputedCacheScores(scorer) if _supports_recompute(scorer)
                else PersistedChunkScores())
    if source not in SLOT_SCORE_SOURCES:
        raise ValueError(
            f"slot score source must be one of {SLOT_SCORE_SOURCE_CHOICES}, "
            f"got {source!r}.")
    if source == RecomputedCacheScores.name:
        if not _supports_recompute(scorer):
            scorer_name = (getattr(scorer, "name", type(scorer).__name__)
                           if scorer is not None else "none")
            raise ValueError(
                f"slot score source 'recompute' needs a scorer that can score "
                f"cached positions (sets rescores_cache and implements "
                f"score_cached), but the active scorer is "
                f"'{scorer_name}'. Use 'auto', or pick a scorer from "
                "scorer.RESCORING_QK_SCORERS.")
        return RecomputedCacheScores(scorer)
    forced_over_recompute = _supports_recompute(scorer)
    if forced_over_recompute:
        logger.warning(
            "KV budget eviction: slot score source forced to 'persist' while "
            "the active scorer can rescore the cache. Cached positions will be "
            "ranked by the score their own chunk produced, which is NOT what "
            "the scorer specifies — this setting is for ablations, not for "
            "serving. Unset --compression-slot-score-source to restore 'auto'.")
    return PersistedChunkScores(forced_over_recompute=forced_over_recompute)
