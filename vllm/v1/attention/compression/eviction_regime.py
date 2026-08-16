# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eviction regime — compression axis 3.

An eviction regime answers three questions for one chunked-prefill step, and
nothing else:

1. **Which cached positions may be evicted this chunk** (the *eval region*)?
2. **How much of that region survives** (the fraction handed to the selection
   level)?
3. **Where do the per-position scores live** between chunks?

The two shipped regimes answer them in opposite ways, which is exactly why they
are separate classes rather than flags on one code path:

* :class:`RatioRegime` (``compression_ratio`` set) — the historical Tangram /
  FastKVzip behaviour. The eval region is CHUNK-LOCAL: only the previous chunk's
  window plus the fresh chunk are re-ranked, and every position promoted to
  "kept" by an earlier chunk is *locked in* (never evicted again). The surviving
  fraction comes from the whole-prompt ratio, so the final cache is
  ``ratio * prompt``, which is only known once the prompt length is. Scores of
  locked positions are never needed again, so the score memory is a small
  chunk-sized workspace.
* :class:`BudgetRegime` (``compression_budget_tokens`` set) — a fixed KV cache
  budget, matching the KeyDiff / H2O / SnapKV reference formulation
  (``if cache_len <= budget: no eviction; else keep the top budget - protected``).
  Eviction fires only once the cache would exceed the budget, and the eval
  region spans the WHOLE cache except the sink and a protected recent tail, so a
  position kept by an earlier chunk can still be evicted later. There is no
  lock-in — the budget is unreachable with it, since a locked prefix only ever
  grows. Because old positions stay rankable, their scores must survive across
  chunks: the regime keeps a persistent per-position statistics buffer that is
  compacted alongside the KV on every eviction.

Both regimes emit the same :class:`ChunkGeometry` + eval-score tensor, so
everything downstream (selection level, position ranking, writeback) is shared
and regime-agnostic.

Terminology used throughout:

* *cache slot* — a position in a (layer, head-group)'s compacted KV, i.e. the
  index the block table and the executor address. Slot 0 is the first sink
  token; slot ``kept_length - 1`` the last live position.
* *member* — one (layer, KV head) pair, row ``layer * num_kv_heads + head``.
* *cluster* — the set of members sharing one head-group's physical KV blocks.
  All members of a cluster necessarily share ONE length, but each keeps its own
  positions inside it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ChunkParams:
    """Per-chunk policy inputs, identical for every layer and group.

    Sourced from ``CompressionRequestMetadata`` (global config forwarded by the
    scheduler) plus the request's own prompt length; a regime reads only the
    fields its formulation needs.
    """
    #: Surviving fraction of the prompt (ratio regime). ``1.0`` means keep all.
    ratio: float
    #: Fixed per-(layer, head-group) KV token budget (budget regime), or None.
    budget_tokens: int | None
    #: Trailing positions never evicted while they are the most recent ones.
    window_size: int
    #: Leading positions never evicted at all.
    n_sink_tokens: int
    #: Whether the chunk just written is itself an eviction candidate
    #: (budget regime only; see ``CacheConfig.compression_evict_current_chunk``).
    evict_current_chunk: bool
    #: Prompt length of this request's first prefill cycle. The ratio regime
    #: needs it to hold one whole-prompt target across chunks; the budget regime
    #: does not (a budget is absolute).
    total_prompt_tokens: int


@dataclass
class ChunkGeometry:
    """Where this chunk's keep decision may act, in cache-slot coordinates.

    Every (layer, group) shares ``sink_size`` and ``tail_size``; the eval region
    starts at ``sink_size + locked[layer, group]`` and is ``real_eval_len[layer,
    group]`` positions long. ``eval_len`` is the maximum of those lengths — the
    rectangular width of the eval-score tensor, which shorter (layer, group)
    pairs pad with ``-inf`` so no padding cell can ever be selected.

    The kept length that follows is
    ``sink_size + locked + <selected> + tail_size``, so the geometry alone fixes
    everything except how many of the eval positions survive.
    """
    #: Leading always-kept positions (never scored, never evicted).
    sink_size: int
    #: Trailing always-kept positions of this chunk. The recent window in the
    #: ratio regime; the whole fresh chunk in the budget regime unless
    #: ``evict_current_chunk`` is set.
    tail_size: int
    #: ``[num_layers, num_groups]`` positions already promoted to permanently
    #: kept by earlier chunks. Always zero in the budget regime (no lock-in).
    locked: torch.Tensor
    #: Rectangular width of the eval-score tensor (max over (layer, group)).
    eval_len: int
    #: ``[num_layers, num_groups]`` genuine eval width per (layer, group); the
    #: selected count is clamped to it so padding is never selected.
    real_eval_len: np.ndarray
    #: Fraction of the eval region to keep, handed to the selection level.
    #: ``>= 1.0`` is the no-eviction fast path, ``<= 0.0`` keeps only the
    #: sink / locked / tail regions.
    adjusted_ratio: float


class RegimeScoreStore(ABC):
    """Per-request score memory owned by one regime.

    A regime decides not only *which* positions may be evicted but also *how
    long their scores must live*, so the two are owned together: the store is
    created by :meth:`EvictionRegime.create_store` and reset with the request.
    """

    @abstractmethod
    def build_eval_scores(
        self,
        pending: torch.Tensor,
        prev_lens: torch.Tensor,
        geometry: ChunkGeometry,
    ) -> torch.Tensor:
        """Absorb this chunk's fresh scores and return the eval-region scores.

        Args:
            pending: ``[num_layers, num_kv_heads, chunk_len]`` scorer output for
                the tokens just written to the cache.
            prev_lens: ``[num_layers, num_groups]`` int64 pre-chunk kept length
                per (layer, group), on the score device.
            geometry: this chunk's geometry.

        Returns:
            ``[num_layers, num_kv_heads, geometry.eval_len]``, padded with the
            dtype's minimum where a (layer, group) has fewer real eval
            positions, so a padding cell can never outrank a real one.
        """

    @abstractmethod
    def compact_cluster(
        self,
        cluster_id: int,
        keep_positions: torch.Tensor,
        kept_length: int,
    ) -> None:
        """Follow one cluster's KV eviction in the score memory.

        Called once per evicted (layer, head-group) with the same
        ``keep_positions`` matrix the KV writeback gathered with, so a
        position's statistics land in the same slot as its KV. Storage for
        evicted positions is released, satisfying the contract that a score
        entry never outlives the KV entry it describes.

        Args:
            cluster_id: flat cluster id (the value ``member_to_cluster`` holds).
            keep_positions: ``[page_group_size, kept_length]`` int64 source cache
                slots per cluster column, in the writeback's column order.
            kept_length: number of live slots after this eviction.
        """

    def note_no_eviction(self) -> None:
        """Hook for a chunk that kept everything (no writeback ran).

        Default: nothing to do — the scores absorbed by
        :meth:`build_eval_scores` already sit at their final slots.
        """


class _ChunkLocalWorkspace(RegimeScoreStore):
    """Score memory for :class:`RatioRegime` — one chunk wide.

    Under lock-in an earlier chunk's kept positions can never be re-ranked, so
    their scores are dead the moment they are locked. The only score that must
    outlive its chunk is the previous chunk's window, which the next chunk
    re-evaluates. So the store is a fixed ``[num_layers, num_kv_heads,
    window + chunk]`` workspace laid out as ``[previous window | fresh chunk]``
    plus that window carry — no growth with prompt length, and no compaction
    work at eviction time.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        # Allocated on the first chunk, once the scorer's dtype / device and the
        # chunk width are known; reused (refilled) by every later chunk.
        self._workspace: torch.Tensor | None = None
        self._workspace_width: int = 0
        # [num_layers, num_kv_heads, win_size] scores of the previous chunk's
        # window, the one region a later chunk still re-ranks.
        self._prior_window: torch.Tensor | None = None
        # Set by the regime before ``build_eval_scores`` — the workspace offset
        # the eval region starts at (the first chunk skips its own sink+window,
        # which occupy no prior-window slots).
        self.eval_start: int = 0

    def build_eval_scores(
        self,
        pending: torch.Tensor,
        prev_lens: torch.Tensor,
        geometry: ChunkGeometry,
    ) -> torch.Tensor:
        del prev_lens  # Chunk-local layout: kept lengths do not address it.
        chunk_len = pending.shape[-1]
        win_size = geometry.tail_size
        width = win_size + chunk_len
        neg_inf = torch.finfo(pending.dtype).min

        if (self._workspace is None
                or self._workspace_width < width
                or self._workspace.dtype != pending.dtype
                or self._workspace.device != pending.device):
            self._workspace = torch.empty(
                self.num_layers, self.num_kv_heads, width,
                dtype=pending.dtype, device=pending.device)
            self._workspace_width = width
        workspace = self._workspace
        workspace.fill_(neg_inf)

        if win_size > 0:
            prior = self._prior_window
            # A carry is usable only if it matches this chunk's window width;
            # ``win_size`` grows on the early chunks of a short prompt, and the
            # first chunk has no carry at all. Leaving the slot at -inf then is
            # correct: there is no earlier window to re-rank.
            if (prior is not None
                    and prior.shape[-1] == win_size
                    and prior.dtype == pending.dtype
                    and prior.device == pending.device):
                workspace[:, :, :win_size] = prior
        workspace[:, :, win_size:width] = pending

        # Carry this chunk's own window into the next chunk. Skipped when
        # nothing is evicted (the no-op path leaves the cache untouched, so no
        # window changes hands).
        if geometry.adjusted_ratio < 1.0:
            if win_size > 0 and chunk_len >= win_size:
                self._prior_window = pending[
                    :, :, chunk_len - win_size:].detach().clone()
            elif win_size == 0:
                self._prior_window = torch.empty(
                    self.num_layers, self.num_kv_heads, 0,
                    dtype=pending.dtype, device=pending.device)
            # A window wider than the chunk is degenerate (config forbids it);
            # leaving the carry untouched keeps the previous chunk's window.

        return workspace[:, :, self.eval_start:self.eval_start + geometry.eval_len]

    def compact_cluster(
        self,
        cluster_id: int,
        keep_positions: torch.Tensor,
        kept_length: int,
    ) -> None:
        # Nothing to compact: the workspace is chunk-local and rebuilt from
        # scratch next chunk, and the window carry is taken from ``pending``
        # (which the eviction does not touch).
        del cluster_id, keep_positions, kept_length


class _PersistentStatBuffer(RegimeScoreStore):
    """Score memory for :class:`BudgetRegime` — one entry per live cache slot.

    Without lock-in every live position stays rankable, so its statistics must
    live exactly as long as its KV does. The buffer is therefore addressed in
    CACHE-SLOT coordinates: ``buffer[layer, head, slot]`` describes whatever
    token that head currently holds in that slot. Two consequences follow, and
    both are load-bearing:

    * a chunk's fresh scores are scattered to ``[prev_len, prev_len + chunk)``,
      where ``prev_len`` is that head's CLUSTER length (cluster members share one
      length, so members of one cluster share the offset);
    * every eviction compacts the buffer with the SAME per-column position
      matrix the KV writeback used, and blanks the slots that fell out, so a
      score entry can never outlive — or drift away from — its KV entry.

    Capacity grows on demand and is bounded by the budget plus one chunk, so it
    does not scale with prompt length. This is the mechanism a statistics-
    accumulating eviction policy (H2O-style running attention mass, recency
    decay, hit counts) plugs into: the slot-aligned lifetime is the hard part,
    and it is solved here once for any such policy.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        member_to_cluster: torch.Tensor,
        cluster_members: torch.Tensor,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        # [num_layers * num_kv_heads] member row -> flat cluster id.
        self.member_to_cluster = member_to_cluster
        # [num_clusters, page_group_size] flat cluster id + column -> member row,
        # the inverse map the compaction needs (the writeback addresses columns,
        # the buffer addresses members).
        self.cluster_members = cluster_members
        self._buffer: torch.Tensor | None = None
        self._capacity: int = 0

    @property
    def buffer(self) -> torch.Tensor | None:
        """The live ``[num_layers, num_kv_heads, capacity]`` statistics, or
        ``None`` before the first chunk. Exposed for tests and for future
        statistics-accumulating policies."""
        return self._buffer

    def _ensure_capacity(
        self,
        needed: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a buffer holding at least ``needed`` slots, preserving the
        statistics already stored. Growth is rare: capacity is only ever
        ``budget + chunk`` in the steady state."""
        neg_inf = torch.finfo(dtype).min
        current = self._buffer
        if (current is not None
                and self._capacity >= needed
                and current.dtype == dtype
                and current.device == device):
            return current
        grown = torch.full(
            (self.num_layers, self.num_kv_heads, needed),
            neg_inf, dtype=dtype, device=device)
        if (current is not None
                and current.dtype == dtype
                and current.device == device):
            grown[:, :, :self._capacity] = current
        self._buffer = grown
        self._capacity = needed
        return grown

    def build_eval_scores(
        self,
        pending: torch.Tensor,
        prev_lens: torch.Tensor,
        geometry: ChunkGeometry,
    ) -> torch.Tensor:
        chunk_len = pending.shape[-1]
        neg_inf = torch.finfo(pending.dtype).min
        # Widest live extent this chunk can produce, before any eviction.
        needed = int(prev_lens.max().item()) + chunk_len
        buffer = self._ensure_capacity(needed, pending.dtype, pending.device)

        # Fresh scores land at each head's own cluster offset. Members of one
        # cluster share its length, so the offset is a per-member lookup into
        # the flattened [num_layers, num_groups] lengths.
        member_offset = prev_lens.reshape(-1)[self.member_to_cluster]
        member_offset = member_offset.view(
            self.num_layers, self.num_kv_heads, 1)
        write_idx = member_offset + torch.arange(
            chunk_len, device=pending.device, dtype=member_offset.dtype
        ).view(1, 1, chunk_len)
        buffer.scatter_(2, write_idx, pending)

        # The eval region starts at ``sink_size`` for every (layer, group) —
        # the budget regime has no locked prefix — and ends where that group's
        # protected tail begins, which differs per group. Read the rectangle and
        # mask the overhang so the protected tail can never be selected.
        eval_len = geometry.eval_len
        if eval_len <= 0:
            return pending.new_full(
                (self.num_layers, self.num_kv_heads, 0), neg_inf)
        sink = geometry.sink_size
        scores = buffer[:, :, sink:sink + eval_len].clone()
        real_len = torch.from_numpy(geometry.real_eval_len).to(
            device=pending.device, dtype=torch.long).reshape(-1)
        member_real_len = real_len[self.member_to_cluster].view(
            self.num_layers, self.num_kv_heads, 1)
        positions = torch.arange(
            eval_len, device=pending.device, dtype=torch.long
        ).view(1, 1, eval_len)
        return scores.masked_fill_(positions >= member_real_len, neg_inf)

    def compact_cluster(
        self,
        cluster_id: int,
        keep_positions: torch.Tensor,
        kept_length: int,
    ) -> None:
        buffer = self._buffer
        if buffer is None:
            return
        flat = buffer.view(self.num_layers * self.num_kv_heads, self._capacity)
        rows = self.cluster_members[cluster_id]
        # Gather first (a fresh tensor), then write back: an in-place gather
        # along a permuted index would read slots it has already overwritten.
        flat[rows, :kept_length] = flat[rows].gather(1, keep_positions)
        # Blank what fell out, so an evicted token's statistics can never be
        # read back by a later chunk (the score entry dies with the KV entry).
        if kept_length < self._capacity:
            flat[rows, kept_length:] = torch.finfo(buffer.dtype).min


class EvictionRegime(ABC):
    """Axis-3 rule: eval region + surviving fraction + score lifetime.

    Stateless — one shared instance per compressor, exactly like a selection
    level. Everything per-request lives in the store the regime creates.
    """

    #: Stable identifier for logging / introspection.
    name: str

    @abstractmethod
    def create_store(
        self,
        num_layers: int,
        num_kv_heads: int,
        member_to_cluster: torch.Tensor,
        cluster_members: torch.Tensor,
    ) -> RegimeScoreStore:
        """Create this regime's per-request score memory."""

    @abstractmethod
    def plan(
        self,
        store: RegimeScoreStore,
        prev_lens: np.ndarray,
        chunk_len: int,
        prev_locked: torch.Tensor,
        is_first_chunk: bool,
        params: ChunkParams,
        device: torch.device,
    ) -> ChunkGeometry:
        """Decide this chunk's geometry.

        Args:
            store: this request's score memory (a regime may pre-arm it here).
            prev_lens: ``[num_layers, num_groups]`` int64 pre-chunk kept lengths.
            chunk_len: tokens written to the cache since the last decision.
            prev_locked: ``[num_layers, num_groups]`` int64 locked counts carried
                from the previous chunk, on ``device``.
            is_first_chunk: whether this is the request's first keep decision.
            params: per-chunk policy inputs.
            device: device the returned ``locked`` tensor must live on.
        """


class RatioRegime(EvictionRegime):
    """Keep a fixed FRACTION of the prompt, with lock-in (historical default).

    The eval region is the previous chunk's window plus the fresh chunk minus
    its own new window; everything a previous chunk promoted is locked in and
    is not re-ranked. The cache therefore only grows, converging on
    ``ratio * prompt`` — which is why the target is derived from the whole
    prompt length rather than from what is currently cached.

    ``adjusted_ratio`` reproduces baseline FastKVzip's window correction
    (``wrapper.py:188-194``): the always-kept window is subtracted from both the
    numerator and denominator, so the fraction applies to the genuinely
    evictable region and the end-to-end retention still lands on ``ratio``.
    """

    name = "ratio"

    def create_store(
        self,
        num_layers: int,
        num_kv_heads: int,
        member_to_cluster: torch.Tensor,
        cluster_members: torch.Tensor,
    ) -> RegimeScoreStore:
        del member_to_cluster, cluster_members  # Chunk-local: no slot mapping.
        return _ChunkLocalWorkspace(num_layers, num_kv_heads)

    def plan(
        self,
        store: RegimeScoreStore,
        prev_lens: np.ndarray,
        chunk_len: int,
        prev_locked: torch.Tensor,
        is_first_chunk: bool,
        params: ChunkParams,
        device: torch.device,
    ) -> ChunkGeometry:
        assert isinstance(store, _ChunkLocalWorkspace)
        total_seen = prev_lens + chunk_len
        min_total = int(total_seen.min())
        sink_size = min(params.n_sink_tokens, min_total)
        win_size = min(params.window_size, max(0, min_total - sink_size))

        # Positions an earlier chunk promoted, clamped to what this chunk's
        # cache can actually hold outside the sink and window.
        max_locked = torch.from_numpy(
            np.maximum(total_seen - sink_size - win_size, 0)
        ).to(device=device, dtype=torch.long)
        locked = torch.minimum(prev_locked.to(device), max_locked)

        adjusted_ratio = self._adjusted_ratio(params, sink_size, win_size)

        # The first chunk has no previous window to re-rank and its own sink is
        # never scored, so its eval region starts past both; later chunks start
        # at the carried window, i.e. at workspace offset 0.
        store.eval_start = win_size + sink_size if is_first_chunk else 0
        eval_len = max(0, chunk_len - store.eval_start)
        real_eval_len = np.full(prev_lens.shape, eval_len, dtype=np.int64)

        return ChunkGeometry(
            sink_size=sink_size,
            tail_size=win_size,
            locked=locked,
            eval_len=eval_len,
            real_eval_len=real_eval_len,
            adjusted_ratio=adjusted_ratio,
        )

    @staticmethod
    def _adjusted_ratio(
        params: ChunkParams,
        sink_size: int,
        win_size: int,
    ) -> float:
        """Baseline FastKVzip's window-corrected fraction. ``win_size`` is held
        fixed (the reference's window-shrink branch instead drops the fraction
        to zero), and the sink is excluded from the prompt it applies to."""
        ratio = params.ratio
        eff_prompt = max(0, int(params.total_prompt_tokens) - sink_size)
        if ratio >= 1.0 or eff_prompt <= win_size:
            return 1.0
        if ratio * eff_prompt < win_size:
            return 0.0
        return max(0.0, min(1.0,
            (ratio * eff_prompt - win_size) / (eff_prompt - win_size)))


class BudgetRegime(EvictionRegime):
    """Hold the cache at a fixed TOKEN BUDGET, with no lock-in.

    Mirrors the reference formulation shared by KeyDiff, H2O and SnapKV: while
    the cache fits the budget nothing is evicted; once a chunk would overflow
    it, the cache is cut back to the budget by keeping the top-scoring positions
    outside the sink and a protected recent tail. Two properties follow that the
    ratio regime does not have:

    * **the target is absolute**, so it holds without knowing the prompt length
      and survives a prompt longer than expected;
    * **nothing is locked in** — a position kept by an earlier chunk competes
      again every chunk, which is the only way a bounded cache can admit later,
      more important tokens. Its score must therefore still be available, which
      is what :class:`_PersistentStatBuffer` provides.

    The protected tail is the whole fresh chunk by default
    (``evict_current_chunk=False``): the reference protects the recent region,
    and keeping the chunk that was just attended over measures better. Setting
    ``evict_current_chunk`` shrinks the protection to the always-kept window, so
    the fresh chunk competes like any other region.

    ``budget_tokens`` is the per-(layer, head-group) length, and — exactly as
    ``compression_ratio`` does — the selection level decides the SCOPE it is
    shared over: ``uniform`` gives every (layer, group) that length, while the
    per-layer / cross-layer levels pool it and let strong groups keep more than
    weak ones at the same total.
    """

    name = "budget"

    def create_store(
        self,
        num_layers: int,
        num_kv_heads: int,
        member_to_cluster: torch.Tensor,
        cluster_members: torch.Tensor,
    ) -> RegimeScoreStore:
        return _PersistentStatBuffer(
            num_layers, num_kv_heads, member_to_cluster, cluster_members)

    def plan(
        self,
        store: RegimeScoreStore,
        prev_lens: np.ndarray,
        chunk_len: int,
        prev_locked: torch.Tensor,
        is_first_chunk: bool,
        params: ChunkParams,
        device: torch.device,
    ) -> ChunkGeometry:
        del store, prev_locked, is_first_chunk  # No lock-in, no first-chunk case.
        assert params.budget_tokens is not None, (
            "BudgetRegime requires compression_budget_tokens.")
        budget = int(params.budget_tokens)
        total_seen = prev_lens + chunk_len
        min_total = int(total_seen.min())
        sink_size = min(params.n_sink_tokens, min_total)
        win_size = min(params.window_size, max(0, min_total - sink_size))
        # Protected tail: the whole fresh chunk, or just the always-kept window
        # when the fresh chunk is allowed to compete. Clamped so it cannot
        # overlap the sink on a cache shorter than the chunk.
        tail_size = win_size if params.evict_current_chunk else min(
            chunk_len, max(0, min_total - sink_size))

        # No lock-in: every position outside sink and tail is a candidate.
        locked = torch.zeros(
            prev_lens.shape, dtype=torch.long, device=device)
        real_eval_len = np.maximum(total_seen - sink_size - tail_size, 0)
        eval_len = int(real_eval_len.max())

        # How much of the eval region may survive. ``budget - sink - tail`` is
        # what the budget leaves for it; expressing that as a fraction of the
        # (rectangular) eval width is what lets every selection level enforce a
        # budget without knowing about budgets: a level keeps that fraction of
        # its scope's cells, which is exactly ``budget - sink - tail`` positions
        # per (layer, group) on average, pooled over whatever scope the level
        # spans. Padding cells hold -inf and so are never among them.
        selectable = budget - sink_size - tail_size
        if eval_len <= 0 or selectable >= eval_len:
            # The cache still fits the budget — nothing to evict this chunk.
            adjusted_ratio = 1.0
        elif selectable <= 0:
            adjusted_ratio = 0.0
        else:
            adjusted_ratio = selectable / eval_len

        return ChunkGeometry(
            sink_size=sink_size,
            tail_size=tail_size,
            locked=locked,
            eval_len=eval_len,
            real_eval_len=real_eval_len.astype(np.int64),
            adjusted_ratio=adjusted_ratio,
        )


#: Axis-3 registry: regime name -> class. Adding a regime is one subclass plus
#: one entry here; nothing else branches on the regime.
_REGIMES: dict[str, type[EvictionRegime]] = {
    RatioRegime.name: RatioRegime,
    BudgetRegime.name: BudgetRegime,
}

#: Valid regime names, so the accepted set lives in one place.
EVICTION_REGIMES: tuple[str, ...] = tuple(_REGIMES)


def make_eviction_regime(regime: str) -> EvictionRegime:
    """Axis-3 dispatch — the ONE place the regime is chosen.

    * ``"ratio"`` — keep ``compression_ratio`` of the prompt, with lock-in.
    * ``"budget"`` — hold the cache at ``compression_budget_tokens``, no lock-in.
    """
    try:
        return _REGIMES[regime]()
    except KeyError:
        raise ValueError(
            f"make_eviction_regime: unknown eviction regime {regime!r}; "
            f"expected one of {EVICTION_REGIMES}.") from None
