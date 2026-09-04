# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eviction regime — compression axis 3.

A regime decides which cached positions may be evicted this chunk (the eval
region), what fraction survives, and how long a score lives. Both emit the same
``ChunkGeometry`` plus eval-score tensor, so the budget scope, the ranking and
the writeback stay regime-agnostic.

The score lifetime is what forces two classes rather than a flag: ``ratio``
locks earlier keeps in, so their scores are dead and one chunk of memory
suffices, while ``budget`` keeps every live position rankable and so must hold a
score for exactly as long as its KV.
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
    cluster_member_rows,
)

if TYPE_CHECKING:
    from vllm.v1.attention.compression.workspace import CompressionWorkspace


@dataclass(frozen=True)
class ChunkParams:
    """Per-chunk policy inputs, identical for every layer and group. From
    ``CompressionRequestMetadata`` plus the request's prompt length; a regime
    reads only the fields its formulation needs.
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
    #: Prompt length of this request's first prefill cycle, needed only by the
    #: ratio regime to hold one whole-prompt target across chunks.
    total_prompt_tokens: int


@dataclass
class ChunkGeometry:
    """Where this chunk's keep decision may act, in cache-slot coordinates.

    Every entry shares ``sink_size`` and ``tail_size``; its eval region starts
    at ``sink_size + locked`` and runs ``real_eval_len`` positions. ``eval_len``
    is the maximum of those -- the rectangular tensor width, which shorter
    entries pad with ``-inf`` so padding is never selected.

    The kept length is ``sink_size + locked + <selected> + tail_size``, so the
    geometry fixes everything but how many eval positions survive.
    """
    #: Leading always-kept positions (never scored, never evicted).
    sink_size: int
    #: Trailing always-kept positions: the recent window (ratio) or the whole
    #: fresh chunk (budget, unless ``evict_current_chunk``).
    tail_size: int
    #: ``[num_layers, num_groups]`` positions already promoted to permanently
    #: kept by earlier chunks. Always zero in the budget regime (no lock-in).
    locked: torch.Tensor
    #: Rectangular width of the eval-score tensor (max over (layer, group)).
    eval_len: int
    #: ``[num_layers, num_groups]`` genuine eval width per (layer, group); the
    #: selected count is clamped to it so padding is never selected.
    real_eval_len: np.ndarray
    #: Fraction of the eval region to keep. ``>= 1.0`` is the no-eviction fast
    #: path, ``<= 0.0`` keeps only sink / locked / tail.
    adjusted_ratio: float


class RegimeScoreStore(ABC):
    """Per-request score memory owned by one regime.

    A regime decides how long a score must live, so it owns the memory holding
    it. Nothing is allocated here: a store is VIEWS into the preallocated
    workspace -- its own row for what outlives a step, shared slabs for what
    does not -- released with the request.
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

        ``[num_layers, num_kv_heads, geometry.eval_len]``, padded with the
        dtype minimum where an entry has fewer real positions so padding cannot
        outrank a real cell. A view into shared memory, valid only until the
        next request's decision this step.
        """

    #: Whether ``compact_cluster`` reads its positions at all. When it does
    #: not, the executor need not materialise them.
    follows_positions: bool = False

    def compaction_target(self) -> "SlotCompactionTarget | None":
        """The score memory the write-back must compact alongside the KV, or
        ``None`` when nothing here outlives the eviction."""
        return None

    @abstractmethod
    def compact_cluster(
        self,
        cluster_id: int,
        keep_positions: torch.Tensor,
        kept_length: int,
    ) -> None:
        """Follow one cluster's KV eviction in the score memory.

        Called once per evicted entry with the same ``keep_positions``
        (``[page_group_size, kept_length]`` int64 source slots per column, in
        the writeback's column order) the KV writeback gathered with, so
        statistics land in the same slot as their KV and evicted storage is
        released: a score entry never outlives the KV entry it describes.
        """


class _ChunkLocalWorkspace(RegimeScoreStore):
    """Score memory for :class:`RatioRegime` — one chunk wide.

    Lock-in kills a kept position's score the moment it is locked, so the only
    score outliving its chunk is the previous window, which the next chunk
    re-evaluates. Hence the shared ``[previous window | fresh chunk]`` slab plus
    the row's carry: no growth with prompt length, no compaction on eviction.
    """

    def __init__(
        self,
        workspace: "CompressionWorkspace",
        row: int,
    ) -> None:
        super().__init__(workspace, row)
        # The window grows over a short prompt's first chunks, and a carry of
        # another width describes other positions, so it is discarded.
        self._carry_width: int = -1
        # Where the eval region starts in the slab, set before
        # ``build_eval_scores``: the first chunk skips its own sink and window.
        self.eval_start: int = 0

    def reset(self) -> None:
        self._carry_width = -1

    def build_eval_scores(
        self,
        inputs: ChunkScoreInputs,
        geometry: ChunkGeometry,
    ) -> torch.Tensor:
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
        # A mismatch, or the first chunk, correctly leaves it unrankable.
        staging[:, :, win_size:width].copy_(pending)

        # Skipped when nothing is evicted: no window changes hands.
        if geometry.adjusted_ratio < 1.0:
            if win_size > 0 and chunk_len >= win_size:
                self.workspace.prior_window[
                    self.row, :, :, :win_size].copy_(
                        pending[:, :, chunk_len - win_size:])
                self._carry_width = win_size
            elif win_size == 0:
                self._carry_width = 0
            # A window wider than the chunk is degenerate; keep the carry.

        return staging[
            :, :, self.eval_start:self.eval_start + geometry.eval_len]

    def compact_cluster(
        self,
        cluster_id: int,
        keep_positions: torch.Tensor,
        kept_length: int,
    ) -> None:
        # Nothing to compact: the slab is rebuilt and the carry untouched.
        del cluster_id, keep_positions, kept_length


class _SlotScoreStore(RegimeScoreStore):
    """Score memory for :class:`BudgetRegime` — one entry per live cache slot.

    Without lock-in every live position stays rankable, so a score must live
    exactly as long as its KV. The buffer is therefore addressed in CACHE-SLOT
    coordinates: ``buffer[layer, head, slot]`` describes whatever token that
    head currently holds there. Two load-bearing consequences:

    * refreshing it each chunk is delegated to a :class:`SlotScoreSource`,
      because the methods disagree on what an old position's score even means;
    * every eviction compacts it with the SAME per-column position matrix the KV
      writeback used and blanks what fell out, so a score entry can neither
      outlive nor drift away from its KV entry.
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
        # A source that keeps history needs its own row, or concurrent
        # requests overwrite each other. Both sides come from one rule.
        if (source.slots_persist_across_steps
                and workspace.spec.slot_rows <= 1):
            raise RuntimeError(
                f"slot score source '{source.name}' keeps a slot's score "
                f"between steps, but the workspace reserved "
                f"{workspace.spec.slot_rows} row(s) for "
                f"{workspace.spec.max_num_reqs} concurrent requests. The "
                "workspace and the compressor were configured from different "
                "compression_scorer / compression_slot_score_source values.")
        # [num_layers, num_kv_heads, slot_capacity]. Shared with every other
        # request in the step unless the source keeps scores between steps.
        self.buffer = workspace.stat_buffer_for(row)
        self.capacity = self.buffer.shape[-1]
        self.source = source
        num_layers, num_kv_heads, _ = self.buffer.shape
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._flat = self.buffer.view(num_layers * num_kv_heads, self.capacity)
        self._member_to_cluster = member_to_cluster
        # On the CPU so the per-cluster loops do not synchronise per cluster.
        self._cluster_members_cpu = cluster_members.cpu().numpy()
        self._num_groups = self._cluster_members_cpu.shape[0] // num_layers

    def reset(self) -> None:
        # Only history needs clearing: a rewriting source leaves nothing
        # readable, and blanking its shared buffer would rob another request.
        if not self.source.slots_persist_across_steps:
            return
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
                f"slot capacity {self.capacity}. The keep decision holds "
                "every (layer, group) to the workspace's per-group capacity, "
                "so this means that ceiling was not applied.")

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

        # Starts at ``sink_size`` for every entry (no locked prefix here),
        # ending where that group's tail begins. Copy the rectangle out -- the
        # buffer keeps the tail scores -- and mask the overhang.
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

    @property
    def follows_positions(self) -> bool:
        # Only a score that will be read again must follow the KV; a
        # recomputing source rebuilds every live slot next eviction anyway.
        return self.source.slots_persist_across_steps

    def compaction_target(self) -> "SlotCompactionTarget | None":
        if not self.follows_positions:
            return None
        from vllm.v1.attention.compression.eviction_writeback import (
            SlotCompactionTarget)
        return SlotCompactionTarget(
            flat=self._flat,
            cluster_members_cpu=self._cluster_members_cpu,
            neg_inf=self.neg_inf)

    def compact_cluster(
        self,
        cluster_id: int,
        keep_positions: torch.Tensor,
        kept_length: int,
    ) -> None:
        if not self.follows_positions:
            return
        rows = cluster_member_rows(
            self._cluster_members_cpu, cluster_id, caller="compact_cluster")
        if rows is None:
            return
        # Not in place: a permuted gather would read what it overwrote.
        self._flat[rows, :kept_length] = self._flat[rows].gather(
            1, keep_positions)
        # An evicted token's score must not be readable by a later chunk.
        if kept_length < self.capacity:
            self._flat[rows, kept_length:] = self.neg_inf


class EvictionRegime(ABC):
    """Axis-3 rule: eval region + surviving fraction + score lifetime.

    Stateless — one shared instance per compressor, exactly like a budget
    scope. Everything per-request lives in the store the regime creates.
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

    #: Whether the slot score source is consumed at all. False for a
    #: chunk-local regime, which has no old position to score.
    uses_slot_scores: bool = False

    def consumes_chunk_scores(self, source: SlotScoreSource) -> bool:
        """Whether the per-chunk scorer must run. True by default: a
        chunk-local eval region has nothing but the chunk's own scores to rank.
        A regime whose source can reconstruct a score from the cache overrides
        this, and the scorer forward is then skipped entirely.
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

        ``prev_lens`` and ``prev_locked`` are the ``[num_layers, num_groups]``
        int64 kept lengths and locked counts the previous chunk left,
        ``chunk_len`` the tokens written since. A regime may pre-arm ``store``
        here. ``device`` is where the returned ``locked`` must live.
        """


class RatioRegime(EvictionRegime):
    """Keep a fixed FRACTION of the prompt, with lock-in.

    The eval region is the previous window plus the fresh chunk minus its own
    new window; everything a previous chunk promoted is locked in. The cache
    therefore only grows, converging on ``keep_ratio * prompt`` -- which is why
    the target comes from the whole prompt length, not from what is cached.

    ``adjusted_ratio`` is baseline FastKVzip's window correction: the always-kept
    window leaves both numerator and denominator, so the fraction applies to the
    genuinely evictable region and end-to-end retention still lands on
    ``keep_ratio``.
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

        # Clamped to what this chunk's cache holds outside sink and window.
        max_locked = torch.from_numpy(
            np.maximum(total_seen - sink_size - win_size, 0)
        ).to(device=device, dtype=torch.long)
        locked = torch.minimum(prev_locked.to(device), max_locked)

        adjusted_ratio = self._adjusted_ratio(params, sink_size, win_size)

        # The first chunk has no previous window and never scores its own
        # sink, so it starts past both; later chunks at offset 0.
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
        """Baseline FastKVzip's window-corrected fraction. ``win_size`` stays
        fixed -- the reference's window-shrink branch drops the fraction to zero
        instead -- and the sink leaves the prompt it applies to."""
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

    The reference formulation shared by KeyDiff, H2O and SnapKV: nothing is
    evicted while the cache fits the budget, and once a chunk would overflow it
    the cache is cut back to the top-scoring positions outside the sink and a
    protected tail. Two properties the ratio regime lacks follow: the target is
    ABSOLUTE, so it holds without knowing the prompt length, and NOTHING is
    locked in -- an earlier keep competes again every chunk, the only way a
    bounded cache can admit later, more important tokens, which is why its score
    must still be available (:class:`_SlotScoreStore`).

    ``budget_tokens`` is the per-entry length; the scope decides the range it is
    shared over, exactly as for ``compression_ratio``.
    """

    name = "budget"
    uses_slot_scores = True

    def consumes_chunk_scores(self, source: SlotScoreSource) -> bool:
        # The region spans earlier chunks, which a recomputing source rebuilds.
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
        # One sink and one tail for every entry, so they can be no wider than
        # the SHORTEST entry holds -- protecting a position that does not exist
        # would put its kept length above what it has seen. Binding here means
        # nothing has been evicted yet: an evicted entry keeps sink + tail, so
        # by the time lengths can differ they are all past this clamp.
        min_total = int(total_seen.min())
        sink_size = min(params.n_sink_tokens, min_total)
        evictable = max(0, min_total - sink_size)
        win_size = min(params.window_size, evictable)
        # The window, widened to the fresh chunk unless it may compete.
        tail_size = win_size if params.evict_current_chunk else min(
            max(win_size, chunk_len), evictable)

        # No lock-in: every position outside sink and tail is a candidate.
        locked = torch.zeros(
            prev_lens.shape, dtype=torch.long, device=device)
        real_eval_len = np.maximum(total_seen - sink_size - tail_size, 0)
        eval_len = int(real_eval_len.max())

        # Expressing what the budget leaves as a FRACTION of the rectangular
        # eval width is what lets a scope enforce a budget without knowing
        # about budgets: that fraction of its cells is ``budget - sink - tail``
        # per entry on average, pooled over whatever it spans.
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
