# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eviction regime — compression axis 3.

An eviction regime answers three questions for one chunked-prefill step, and
nothing else:

1. **Which cached positions may be evicted this chunk** (the *eval region*)?
2. **How much of that region survives** (the fraction handed to the budget
   scope)?
3. **Where do the per-position scores live** between chunks?

The two shipped regimes answer them in opposite ways, which is exactly why they
are separate classes rather than flags on one code path:

* :class:`RatioRegime` (``compression_ratio`` set) — the historical Tangram /
  FastKVzip behaviour. The eval region is CHUNK-LOCAL: only the previous chunk's
  window plus the fresh chunk are re-ranked, and every position promoted to
  "kept" by an earlier chunk is *locked in* (never evicted again). The surviving
  fraction comes from the whole-prompt keep ratio, so the final cache is
  ``keep_ratio * prompt``, which is only known once the prompt length is. Scores of
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
everything downstream (budget scope, position ranking, writeback) is shared
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
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.v1.attention.compression.slot_scores import (
    ChunkScoreInputs,
    SlotFillTarget,
    SlotScoreSource,
)

if TYPE_CHECKING:
    from vllm.v1.attention.compression.workspace import CompressionWorkspace


@dataclass(frozen=True)
class ChunkParams:
    """Per-chunk policy inputs, identical for every layer and group.

    Sourced from ``CompressionRequestMetadata`` (global config forwarded by the
    scheduler) plus the request's own prompt length; a regime reads only the
    fields its formulation needs.
    """
    #: Surviving fraction of the prompt (ratio regime): ``1 -
    #: CacheConfig.compression_ratio``. ``1.0`` means keep all.
    keep_ratio: float
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
    #: Fraction of the eval region to keep, handed to the budget scope.
    #: ``>= 1.0`` is the no-eviction fast path, ``<= 0.0`` keeps only the
    #: sink / locked / tail regions.
    adjusted_ratio: float


class RegimeScoreStore(ABC):
    """Per-request score memory owned by one regime.

    A regime decides not only *which* positions may be evicted but also *how
    long their scores must live*, so the two are owned together. The memory
    itself is never allocated here: a store is a set of VIEWS into the
    preallocated :class:`CompressionWorkspace` (its own row for the parts that
    outlive a step, the shared slabs for the parts that do not), created by
    :meth:`EvictionRegime.create_store` and released with the request.
    """

    def __init__(
        self,
        workspace: "CompressionWorkspace",
        row: int,
    ) -> None:
        self.workspace = workspace
        self.row = row
        self.neg_inf = float(
            torch.finfo(workspace.spec.score_dtype).min)

    @abstractmethod
    def reset(self) -> None:
        """Clear the row's state so nothing leaks from the previous request that
        occupied it (rows are recycled)."""

    @abstractmethod
    def build_eval_scores(
        self,
        inputs: ChunkScoreInputs,
        geometry: ChunkGeometry,
    ) -> torch.Tensor:
        """Bring the score memory up to date and return the eval-region scores.

        Args:
            inputs: this chunk's scorer output, pre-chunk lengths and (when the
                regime's score source rescores the cache) read access to it.
            geometry: this chunk's geometry.

        Returns:
            ``[num_layers, num_kv_heads, geometry.eval_len]``, padded with the
            dtype's minimum where a (layer, group) has fewer real eval
            positions, so a padding cell can never outrank a real one. The
            returned tensor is a view into shared workspace memory and is only
            valid until the next request's decision in the same step.
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


class _ChunkLocalWorkspace(RegimeScoreStore):
    """Score memory for :class:`RatioRegime` — one chunk wide.

    Under lock-in an earlier chunk's kept positions can never be re-ranked, so
    their scores are dead the moment they are locked. The only score that must
    outlive its chunk is the previous chunk's window, which the next chunk
    re-evaluates. So this store needs just the shared ``[previous window |
    fresh chunk]`` staging slab plus its row's window carry — no growth with
    prompt length, and no compaction work at eviction time.
    """

    def __init__(
        self,
        workspace: "CompressionWorkspace",
        row: int,
    ) -> None:
        super().__init__(workspace, row)
        # Width of the window the carry was written for. The window grows over
        # the first chunks of a short prompt; a carry written at a different
        # width describes different positions, so it is discarded rather than
        # reinterpreted.
        self._carry_width: int = -1
        # Workspace offset the eval region starts at, set by the regime before
        # ``build_eval_scores`` (the first chunk skips its own sink + window,
        # which no previous window occupies).
        self.eval_start: int = 0

    def reset(self) -> None:
        self._carry_width = -1

    def build_eval_scores(
        self,
        inputs: ChunkScoreInputs,
        geometry: ChunkGeometry,
    ) -> torch.Tensor:
        # Chunk-local layout: kept lengths do not address it, and there is
        # nothing cached to rescore.
        pending = inputs.pending
        chunk_len = pending.shape[-1]
        win_size = geometry.tail_size
        width = win_size + chunk_len
        staging = self.workspace.staging
        if width > staging.shape[-1]:
            raise RuntimeError(
                f"RatioRegime: window + chunk ({width}) exceeds the reserved "
                f"staging width ({staging.shape[-1]}).")

        staging[:, :, :width].fill_(self.neg_inf)
        if win_size > 0 and self._carry_width == win_size:
            staging[:, :, :win_size].copy_(
                self.workspace.prior_window[self.row, :, :, :win_size])
        # A width mismatch (or the first chunk) leaves the window slot at the
        # dtype minimum, which is correct: there is no earlier window to rank.
        staging[:, :, win_size:width].copy_(pending)

        # Carry this chunk's own window into the next chunk. Skipped when
        # nothing is evicted (the no-op path leaves the cache untouched, so no
        # window changes hands).
        if geometry.adjusted_ratio < 1.0:
            if win_size > 0 and chunk_len >= win_size:
                self.workspace.prior_window[
                    self.row, :, :, :win_size].copy_(
                        pending[:, :, chunk_len - win_size:])
                self._carry_width = win_size
            elif win_size == 0:
                self._carry_width = 0
            # A window wider than the chunk is degenerate (config forbids it);
            # leaving the carry untouched keeps the previous chunk's window.

        return staging[
            :, :, self.eval_start:self.eval_start + geometry.eval_len]

    def compact_cluster(
        self,
        cluster_id: int,
        keep_positions: torch.Tensor,
        kept_length: int,
    ) -> None:
        # Nothing to compact: the staging slab is rebuilt from scratch next
        # chunk, and the window carry is taken from ``pending`` (which the
        # eviction does not touch).
        del cluster_id, keep_positions, kept_length


class _SlotScoreStore(RegimeScoreStore):
    """Score memory for :class:`BudgetRegime` — one entry per live cache slot.

    Without lock-in every live position stays rankable, so a score must be
    available for it exactly as long as its KV is. The row's buffer is therefore
    addressed in CACHE-SLOT coordinates: ``buffer[layer, head, slot]`` describes
    whatever token that head currently holds in that slot. Two consequences
    follow, and both are load-bearing:

    * bringing the buffer up to date each chunk is delegated to a
      :class:`SlotScoreSource`, because the eviction methods disagree about what
      "the score of an old position" even means — recompute it from the cached
      keys (KeyDiff), accumulate it over the queries seen so far (H2O), or keep
      what its own chunk produced (chunk-local scorers). See ``slot_scores.py``;
    * every eviction compacts the buffer with the SAME per-column position
      matrix the KV writeback used, and blanks the slots that fell out, so a
      score entry can never outlive — or drift away from — its KV entry.
    """

    def __init__(
        self,
        workspace: "CompressionWorkspace",
        row: int,
        member_to_cluster: torch.Tensor,
        cluster_members: torch.Tensor,
        source: SlotScoreSource,
    ) -> None:
        super().__init__(workspace, row)
        if workspace.stat_buffer is None:
            raise RuntimeError(
                "BudgetRegime needs the per-slot score buffer, but the "
                "workspace was built without one (slot_capacity == 0).")
        # [num_layers, num_kv_heads, slot_capacity] — this row's slice.
        self.buffer = workspace.stat_buffer[row]
        self.capacity = self.buffer.shape[-1]
        self.source = source
        num_layers, num_kv_heads, _ = self.buffer.shape
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._flat = self.buffer.view(num_layers * num_kv_heads, self.capacity)
        self._member_to_cluster = member_to_cluster
        # The per-cluster loops address rows from the CPU so they do not
        # synchronise on the device once per cluster.
        self._cluster_members_cpu = cluster_members.cpu().numpy()
        self._num_groups = self._cluster_members_cpu.shape[0] // num_layers

    def reset(self) -> None:
        self.buffer.fill_(self.neg_inf)

    def build_eval_scores(
        self,
        inputs: ChunkScoreInputs,
        geometry: ChunkGeometry,
    ) -> torch.Tensor:
        needed = int(inputs.prev_lens_cpu.max()) + inputs.chunk_len
        if needed > self.capacity:
            raise RuntimeError(
                f"BudgetRegime: live cache length {needed} exceeds the reserved "
                f"slot capacity {self.capacity}. The keep decision caps every "
                "(layer, group) at the budget, so this means the cap was not "
                "applied.")

        self.source.fill(
            SlotFillTarget(
                buffer=self.buffer,
                flat=self._flat,
                member_to_cluster=self._member_to_cluster,
                cluster_members_cpu=self._cluster_members_cpu,
                num_layers=self._num_layers,
                num_kv_heads=self._num_kv_heads,
                num_groups=self._num_groups,
                neg_inf=self.neg_inf,
            ),
            inputs,
        )

        # The eval region starts at ``sink_size`` for every (layer, group) —
        # the budget regime has no locked prefix — and ends where that group's
        # protected tail begins, which differs per group. Copy the rectangle out
        # (the buffer itself must keep the tail scores) and mask the overhang so
        # the protected tail can never be selected.
        eval_len = geometry.eval_len
        out = self.workspace.eval_scores[:, :, :eval_len]
        if eval_len == 0:
            return out
        sink = geometry.sink_size
        out.copy_(self.buffer[:, :, sink:sink + eval_len])
        real_len = torch.from_numpy(geometry.real_eval_len).to(
            device=out.device, dtype=torch.long).reshape(-1)
        member_real_len = real_len[self._member_to_cluster].view(
            self._num_layers, self._num_kv_heads, 1)
        positions = torch.arange(
            eval_len, device=out.device, dtype=torch.long
        ).view(1, 1, eval_len)
        return out.masked_fill_(positions >= member_real_len, self.neg_inf)

    def compact_cluster(
        self,
        cluster_id: int,
        keep_positions: torch.Tensor,
        kept_length: int,
    ) -> None:
        rows = self._cluster_members_cpu[cluster_id]
        if (rows < 0).any():
            if (rows < 0).all():
                return  # Empty cluster: no member holds these slots.
            # Score slots follow KV slots per column, so a member left behind
            # here would keep scores the eviction has already discarded.
            raise RuntimeError(
                f"compact_cluster: cluster {cluster_id} holds members in some "
                f"columns but not others ({rows.tolist()}); a cluster map must "
                "leave a cluster either full or empty.")
        # Gather first (a fresh tensor), then write back: an in-place gather
        # along a permuted index would read slots it has already overwritten.
        self._flat[rows, :kept_length] = self._flat[rows].gather(
            1, keep_positions)
        # Blank what fell out, so an evicted token's score can never be read
        # back by a later chunk (the score entry dies with the KV entry).
        if kept_length < self.capacity:
            self._flat[rows, kept_length:] = self.neg_inf


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
        workspace: "CompressionWorkspace",
        row: int,
        member_to_cluster: torch.Tensor,
        cluster_members: torch.Tensor,
        slot_score_source: SlotScoreSource,
    ) -> RegimeScoreStore:
        """Bind this regime's score memory to one reserved workspace row."""

    #: Whether the regime keeps a slot-addressed score buffer, i.e. whether the
    #: slot score source (``slot_scores.py``) is consumed at all. False for a
    #: chunk-local regime, which has no old position to score.
    uses_slot_scores: bool = False

    def consumes_chunk_scores(self, source: SlotScoreSource) -> bool:
        """Whether the per-chunk scorer must run for this regime.

        Always true by default: a chunk-local eval region has nothing but the
        chunk's own scores to rank. A regime whose score source can reconstruct
        a position's score from the cache overrides this, and the scorer's
        forward pass is then skipped entirely.
        """
        del source
        return True

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
    """Keep a fixed FRACTION of the prompt, with lock-in.

    The eval region is the previous chunk's window plus the fresh chunk minus
    its own new window; everything a previous chunk promoted is locked in and
    is not re-ranked. The cache therefore only grows, converging on
    ``keep_ratio * prompt`` — which is why the target is derived from the whole
    prompt length rather than from what is currently cached.

    ``adjusted_ratio`` reproduces baseline FastKVzip's window correction: the
    always-kept window is subtracted from both the numerator and the
    denominator, so the fraction applies to the genuinely evictable region and
    the end-to-end retention still lands on ``keep_ratio``.
    """

    name = "ratio"

    def create_store(
        self,
        workspace: "CompressionWorkspace",
        row: int,
        member_to_cluster: torch.Tensor,
        cluster_members: torch.Tensor,
        slot_score_source: SlotScoreSource,
    ) -> RegimeScoreStore:
        # Chunk-local: no slot mapping, and no old position to score.
        del member_to_cluster, cluster_members, slot_score_source
        return _ChunkLocalWorkspace(workspace, row)

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
        keep = params.keep_ratio
        eff_prompt = max(0, int(params.total_prompt_tokens) - sink_size)
        if keep >= 1.0 or eff_prompt <= win_size:
            return 1.0
        if keep * eff_prompt < win_size:
            return 0.0
        return max(0.0, min(1.0,
            (keep * eff_prompt - win_size) / (eff_prompt - win_size)))


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
      is what :class:`_SlotScoreStore` and its score source provide.

    ``budget_tokens`` is the per-(layer, head-group) length; the selection
    level decides the scope it is shared over, exactly as it does for
    ``compression_ratio``. See ``CacheConfig.compression_budget_tokens`` and
    ``CacheConfig.compression_evict_current_chunk`` for the user-facing terms.
    """

    name = "budget"
    uses_slot_scores = True

    def consumes_chunk_scores(self, source: SlotScoreSource) -> bool:
        # Under a budget the eval region spans positions from earlier chunks, so
        # what the chunk's own scorer produced is only one of several ways to
        # score them — and a source that recomputes from the cached keys does not
        # need it at all.
        return source.needs_chunk_scores

    def create_store(
        self,
        workspace: "CompressionWorkspace",
        row: int,
        member_to_cluster: torch.Tensor,
        cluster_members: torch.Tensor,
        slot_score_source: SlotScoreSource,
    ) -> RegimeScoreStore:
        return _SlotScoreStore(
            workspace, row, member_to_cluster, cluster_members,
            slot_score_source)

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
        evictable = max(0, min_total - sink_size)
        win_size = min(params.window_size, evictable)
        # Protected tail: the recent window always, widened to the whole fresh
        # chunk unless that chunk is allowed to compete.
        tail_size = win_size if params.evict_current_chunk else min(
            max(win_size, chunk_len), evictable)

        # No lock-in: every position outside sink and tail is a candidate.
        locked = torch.zeros(
            prev_lens.shape, dtype=torch.long, device=device)
        real_eval_len = np.maximum(total_seen - sink_size - tail_size, 0)
        eval_len = int(real_eval_len.max())

        # How much of the eval region may survive. ``budget - sink - tail`` is
        # what the budget leaves for it; expressing that as a fraction of the
        # (rectangular) eval width is what lets every budget scope enforce a
        # budget without knowing about budgets: a scope keeps that fraction of
        # its cells, which is exactly ``budget - sink - tail`` positions per
        # (layer, group) on average, pooled over whatever range the scope
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



def make_eviction_regime(regime: str) -> EvictionRegime:
    """Axis-3 dispatch — the ONE place the regime is chosen.

    * ``"ratio"`` — evict ``compression_ratio`` of the prompt, with lock-in.
    * ``"budget"`` — hold the cache at ``compression_budget_tokens``, no lock-in.
    """
    try:
        return _REGIMES[regime]()
    except KeyError:
        raise ValueError(
            f"make_eviction_regime: unknown eviction regime {regime!r}; "
            f"expected one of {tuple(_REGIMES)}.") from None
