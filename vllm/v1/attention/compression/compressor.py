# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep-decision logic for KV cache compression.

Owns the keep decision and holds no policy of its own: the budget scope
(axis 1), the scorer (axis 2) and the eviction regime (axis 3) supply all
three. KV writes and block-table updates live in the FlashAttention backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np
import torch
from torch import nn

from vllm.logger import init_logger
from vllm.v1.attention.backends.ragged_layout import (
    identity_member_maps,
    load_cluster_map,
    member_maps_from_cluster_map,
)
from vllm.v1.attention.compression.eviction_regime import (
    ChunkParams,
    EvictionRegime,
    RegimeScoreStore,
    make_eviction_regime,
)
from vllm.v1.attention.compression.gate import load_gates
from vllm.v1.attention.compression.gate_capture import (
    _wrap_forward_with_gate_capture,
)
from vllm.v1.attention.compression.budget_scope import (
    BudgetScope,
    make_budget_scope,
)
from vllm.v1.attention.compression.slot_scores import (
    SLOT_SCORE_SOURCE_AUTO,
    ChunkScoreInputs,
    KVCacheView,
    PersistedChunkScores,
    SlotScoreSource,
    make_slot_score_source,
)
from vllm.v1.attention.compression.workspace import CompressionWorkspace
from vllm.v1.attention.compression.scorer import build_qk_scorer

logger = init_logger(__name__)


class KeepDecisionObserver(Protocol):
    """Notified of each finalized per-request keep decision, by offline
    tooling only. With ``page_group_size = 1`` every entry is one head, so
    :meth:`record` receives per-(layer, head) ``kept_lengths`` after eviction
    and ``total_seen`` before it, plus the widths that decided the difference.
    """

    def record(
        self,
        req_id: str,
        *,
        kept_lengths: np.ndarray,
        total_seen: np.ndarray,
        sink_size: int,
        win_size: int,
        eval_len: int,
    ) -> None:
        ...


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


@dataclass
class _RequestCompressState:
    """Per-request bookkeeping: the workspace row that addresses this
    request's tensors, plus the CPU-side results the scheduler and executor
    read back."""
    row: int
    #: Regime-owned score memory: chunk-local staging (ratio) or a slot-aligned
    #: persistent buffer (budget).
    score_store: RegimeScoreStore
    cross_layer_decision: KeepDecision | None = None
    #: Set once an eviction committed a length, so the next chunk is not first.
    has_committed: bool = False
    #: [L, G, page_group_size, eval_len] view into the SHARED ranking buffer.
    #: Borrowed: valid only between this request's ranking and its writeback,
    #: which clears it, because the next request to rank overwrites it.
    borrowed_sorted_indices: torch.Tensor | None = None
    cached_k_new_cpu: np.ndarray | None = None           # [L, G]
    locked_count_cpu: np.ndarray | None = None           # [L, G]
    #: [L, G] genuine eval width. Ragged under the budget regime, where the
    #: score tensor is padded, so the selected count is clamped to this.
    real_eval_len_cpu: np.ndarray | None = None
    #: [L, G] int32 post-evict kept_lengths, MAX-reduced across ranks under TP
    #: for block-pool consistency.
    cached_kept_lengths_cpu: np.ndarray | None = None


def _apportion_blocks(
    want: np.ndarray,
    base: np.ndarray,
    remainder: np.ndarray,
    total: int,
    block_size: int,
) -> np.ndarray:
    """Share one pooled ``total`` over a span's (layer, group) entries.

    A pooled scope hands out uneven counts on purpose, but the selection must
    be block quantized and the span's total must hold. Capping each entry
    separately satisfies both and destroys the unevenness, so quantization is
    per entry and the budget per span.

    Largest-remainder apportionment: each entry takes the block FLOOR of its
    demand (``base``, from ``want``), then the leftover blocks go to whoever
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


class KVCompressor:
    """One instance per model; per-request state held in ``req_state``.

    ``compress_active`` is flipped by the ModelRunner around the compress
    forward and read by the per-layer scorers.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        page_group_size: int,
        head_size: int,
        hidden_dim: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
        workspace: CompressionWorkspace,
        budget_scope: str = "layer",
        regime: str = "ratio",
        slot_score_source: str = SLOT_SCORE_SOURCE_AUTO,
    ) -> None:
        assert num_kv_heads % page_group_size == 0, (
            f"num_kv_heads ({num_kv_heads}) must be divisible by "
            f"page_group_size ({page_group_size}).")

        # Axes 1 and 3, chosen once; nothing downstream branches on either.
        self.scope: BudgetScope = make_budget_scope(budget_scope)
        # The per-entry ceiling IS the reserved width, and a pooling scope
        # needs room for one entry outgrowing ``budget``. Sized for the wrong
        # scope nothing raises: it just caps every entry again, silently.
        if (workspace.spec.slot_capacity > 0
                and workspace.spec.budget_scope != budget_scope):
            raise RuntimeError(
                f"KVCompressor: budget scope '{budget_scope}' does not match "
                f"the workspace, which was sized for "
                f"'{workspace.spec.budget_scope}' "
                f"(per_group_capacity={workspace.spec.per_group_capacity}). "
                "Both come from compression_budget_scope, so they were built "
                "from different configurations.")
        self.regime: EvictionRegime = make_eviction_regime(regime)
        self.workspace = workspace
        self.num_layers = num_layers
        self.num_kv_heads_per_layer = num_kv_heads
        self.page_group_size = page_group_size
        self.num_head_groups_per_layer = num_kv_heads // page_group_size
        self.head_size = head_size
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) \
            else device

        # Installed later, so a unit test needs no scorer.
        # ``scorer_consumes`` decides delivery: ``"qk"`` on the inner
        # ``Attention``, ``"hidden_states"`` wrapping the outer block.
        self.scorers: list[nn.Module] = []
        self.scorer_consumes: str = "hidden_states"
        # Replaced once the scorer is installed: it decides rescorability.
        self.slot_score_source_choice = slot_score_source
        self.slot_score_source: SlotScoreSource = PersistedChunkScores()

        # member -> (cluster, column), both filled by ``set_cluster_map``.
        self.member_to_cluster: torch.Tensor | None = None
        self.member_to_col: torch.Tensor | None = None
        # [num_clusters, page_group_size] inverse, -1 for an unfilled column.
        # The writeback addresses columns, a score buffer members.
        self.cluster_members: torch.Tensor | None = None

        # ``None`` in production, so the keep-decision path does no extra work.
        self.keep_decision_observer: KeepDecisionObserver | None = None

        self.req_state: dict[str, _RequestCompressState] = {}

        # A recomputing source never reads the chunk's own scores, so both the
        # scorer forward and its buffer write are skipped.
        self.compress_active: bool = False
        # ``(req_id, start, end)`` in hidden_states; anything outside is skipped.
        self.pending_req_offsets: list[tuple[str, int, int]] | None = None
        # req_id -> global position of its chunk's first scored token, read by
        # the position-dependent scorers. Set and cleared with the offsets.
        self.pending_req_pos_offsets: dict[str, int] | None = None

    def load_gate_checkpoint(
        self,
        model_name: str,
        gate_path: str,
        num_kv_heads_total: int,
        tp_rank: int,
    ) -> None:
        """Load per-layer gates from a Fast-KVzip checkpoint.

        Under TP, ``num_kv_heads_total`` is the model-global KV-head count
        the checkpoint was trained against; the loader shards it to this
        rank's slice ``[tp_rank * per_rank, (tp_rank + 1) * per_rank)``.
        """
        self.scorers = load_gates(
            model_name=model_name,
            gate_path=gate_path,
            num_layers=self.num_layers,
            num_kv_heads_per_rank=self.num_kv_heads_per_layer,
            num_kv_heads_total=num_kv_heads_total,
            tp_rank=tp_rank,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.scorer_consumes = "hidden_states"
        self._select_slot_score_source(
            self.scorers[0] if self.scorers else None)

    def set_qk_scorers(
        self,
        scorer_name: str,
        num_q_per_kv: int,
        options: Mapping[str, str] | None = None,
    ) -> None:
        """Install a gate-free query/key scorer: one shared stateless instance
        across every compressible layer, scoring from post-RoPE query/key, so no
        checkpoint is loaded.

        ``options`` carries whatever the selected scorer declared in its
        ``OPTIONS``, so this signature does not grow when a scorer gains a
        hyperparameter. The scorer comes from ``build_qk_scorer``, a registry
        lookup with no per-scorer branch, and ``scorer_consumes`` is read off
        the module so ``attach_scorers`` stays scorer-agnostic."""
        scorer = build_qk_scorer(
            scorer_name,
            num_kv_heads=self.num_kv_heads_per_layer,
            num_q_per_kv=num_q_per_kv,
            head_size=self.head_size,
            options=options,
        ).to(device=self.device)
        scorer.eval()
        # Stateless, so one instance; the length matches ``attach_scorers``.
        self.scorers = [scorer for _ in range(self.num_layers)]
        self.scorer_consumes = scorer.consumes
        self._select_slot_score_source(scorer)

    def _select_slot_score_source(self, scorer: nn.Module | None) -> None:
        """Bind the score source the installed scorer supports, and report it
        when the active regime actually consumes it."""
        self.slot_score_source = make_slot_score_source(
            scorer, self.slot_score_source_choice)
        if self.regime.uses_slot_scores:
            logger.info("KV budget eviction: %s.",
                        self.slot_score_source.describe())

    def set_cluster_map(self, head_group_cluster_map: str | None) -> None:
        """Bind the member->(cluster, column) maps the keep decision uses.

        ``None`` selects the identity map; a loaded map assigns each head to
        the possibly cross-layer cluster whose physical blocks it shares. Either
        way these mirror the maps the FlashAttention builder pages with, so
        scoring and physical placement agree.
        """
        if head_group_cluster_map is None:
            member_to_cluster, member_to_col = identity_member_maps(
                self.num_layers,
                self.num_kv_heads_per_layer,
                self.page_group_size,
                self.device,
            )
        else:
            cluster_of, column_of = load_cluster_map(
                head_group_cluster_map,
                self.page_group_size,
                self.num_kv_heads_per_layer,
            )
            if cluster_of.shape[0] != self.num_layers:
                raise ValueError(
                    f"head_group_cluster_map has {cluster_of.shape[0]} layers "
                    f"but the compressor spans {self.num_layers}.")
            member_to_cluster, member_to_col = member_maps_from_cluster_map(
                cluster_of.to(self.device), column_of.to(self.device))
        self.member_to_cluster = member_to_cluster
        self.member_to_col = member_to_col
        num_clusters = self.num_layers * self.num_head_groups_per_layer
        # -1 so a score buffer never mistakes an empty slot for member 0.
        cluster_members = torch.full(
            (num_clusters, self.page_group_size), -1,
            dtype=torch.long, device=self.device)
        cluster_members[member_to_cluster, member_to_col] = torch.arange(
            member_to_cluster.numel(), dtype=torch.long, device=self.device)
        self.cluster_members = cluster_members
        logger.info(
            "KVCompressor scoring cluster map: %s (%d clusters over %d "
            "members, page_group_size=%d)",
            "identity (adjacent-head)" if head_group_cluster_map is None
            else head_group_cluster_map,
            num_clusters,
            self.num_layers * self.num_kv_heads_per_layer,
            self.page_group_size,
        )

    def begin_request(self, req_id: str) -> None:
        if req_id in self.req_state:
            raise RuntimeError(
                f"KVCompressor.begin_request: '{req_id}' already active.")
        if (self.member_to_cluster is None or self.cluster_members is None):
            raise RuntimeError(
                "KVCompressor.begin_request: cluster maps are unset — "
                "set_cluster_map must run after construction.")
        row = self.workspace.acquire_row()
        store = self.regime.create_store(
            self.workspace, row, self.member_to_cluster, self.cluster_members,
            self.slot_score_source)
        store.reset()
        self.req_state[req_id] = _RequestCompressState(row=row,
                                                       score_store=store)

    def end_request(self, req_id: str) -> None:
        # Idempotent for shutdown paths; releasing is what stops a row leak.
        state = self.req_state.pop(req_id, None)
        if state is not None:
            self.workspace.release_row(state.row)

    def receive_score(
        self,
        req_id: str,
        layer_idx: int,
        score: torch.Tensor,
    ) -> None:
        """Stash one layer's ``[num_kv_heads_per_layer, sub_chunk_len]`` score,
        accumulating across budget-sliced sub-chunks until the boundary step
        consumes it. Source-agnostic: gate and qk scorers feed the same buffer.

        Concurrency lets the scheduler split one compression chunk over several
        forward steps, each scoring only its own tokens; concatenating in
        arrival order assembles a full ``chunk_size`` by the boundary step,
        where ``_take_pending`` consumes and rewinds it.

        This is byte-equivalent to a serial full-chunk score for the
        hidden-states gate only, whose per-token score does not depend on how
        prefill was sliced. A qk scorer measures a chunk-relative observation
        window, so a sub-chunk's window is not the full chunk's and the
        assembled buffer differs from the serial one -- accepted, the
        alternative being to refuse to score until a chunk is whole. Do not use
        bit-identical output as a test oracle for a chunk-relative scorer."""
        if score.shape[0] != self.num_kv_heads_per_layer:
            raise ValueError(
                f"score head dim {score.shape[0]} != "
                f"num_kv_heads_per_layer {self.num_kv_heads_per_layer}.")
        state = self.req_state.get(req_id)
        if state is None:
            raise RuntimeError(
                f"KVCompressor.receive_score: '{req_id}' not "
                "begin_request'd.")
        workspace = self.workspace
        cursor = int(workspace.pending_len[state.row, layer_idx])
        sub_chunk_len = score.shape[1]
        end = cursor + sub_chunk_len
        if end > workspace.pending_score.shape[-1]:
            raise RuntimeError(
                f"KVCompressor.receive_score(layer={layer_idx}): scores for "
                f"{end} tokens exceed the reserved chunk width "
                f"{workspace.pending_score.shape[-1]}; a compression chunk "
                "must never exceed compression_chunk_size.")
        if score.dtype != workspace.spec.score_dtype:
            raise RuntimeError(
                f"KVCompressor.receive_score(layer={layer_idx}): scorer "
                f"produced {score.dtype} but the workspace reserved "
                f"{workspace.spec.score_dtype}. The reserved dtype is derived "
                "from compression_scorer and must match it exactly (rounding "
                "would change the keep decision).")
        workspace.pending_score[
            state.row, layer_idx, :, cursor:end].copy_(score)
        workspace.pending_len[state.row, layer_idx] = end

    @property
    def chunk_scoring_enabled(self) -> bool:
        """Whether the per-layer scorer has to run this step. False only when the
        active regime reconstructs a position's score from the cache instead of
        from the chunk that wrote it, in which case scoring the chunk would be
        pure waste."""
        return self.regime.consumes_chunk_scores(self.slot_score_source)

    def prepare_keep_decision(
        self,
        req_id: str,
        prev_seq_lens_per_layer: torch.Tensor,
        chunk_len: int,
        params: ChunkParams,
        cache_view: KVCacheView | None = None,
    ) -> KeepDecision:
        """Run the keep decision for one chunk.

        Three delegations: the regime fixes the geometry and supplies the
        eval-region scores, the scope turns those into a per-entry kept COUNT,
        and this caches the POSITION ranking the executor gathers with.
        ``prev_seq_lens_per_layer`` must equal the last committed lengths, or be
        all-zero on the first chunk.
        """
        if req_id not in self.req_state:
            raise RuntimeError(
                f"prepare_keep_decision: '{req_id}' not begin_request'd.")
        if not (0.0 < params.keep_ratio <= 1.0):
            raise ValueError(
                f"prepare_keep_decision: keep_ratio must be in (0, 1], got "
                f"{params.keep_ratio}.")

        req = self.req_state[req_id]
        num_layers = self.num_layers
        num_groups = self.num_head_groups_per_layer
        num_kv_heads = self.num_kv_heads_per_layer

        prev_lens = prev_seq_lens_per_layer.to(dtype=torch.long).cpu()
        if prev_lens.shape != (num_layers, num_groups):
            raise ValueError(
                f"prev_seq_lens shape {tuple(prev_lens.shape)} != "
                f"({num_layers}, {num_groups}).")

        self._assert_once_only(req, prev_lens, num_layers, num_groups)

        # One [L, num_kv_heads, chunk_len] view; consuming rewinds the cursors.
        pending = self._take_pending(req, num_layers, chunk_len)
        device = pending.device
        store = req.score_store

        geometry = self.regime.plan(
            store=store,
            prev_lens=prev_lens.numpy(),
            chunk_len=chunk_len,
            prev_locked=self.workspace.locked[req.row],
            is_first_chunk=not req.has_committed,
            params=params,
            device=device,
        )
        sink_size = geometry.sink_size
        eval_len = geometry.eval_len
        adjusted_ratio = geometry.adjusted_ratio
        locked = geometry.locked
        if eval_len > self.workspace.spec.eval_capacity:
            raise RuntimeError(
                f"prepare_keep_decision: eval region {eval_len} exceeds the "
                f"reserved capacity {self.workspace.spec.eval_capacity}; the "
                "workspace is sized from the regime's own bound, so this means "
                "the two disagree.")

        eval_scores = store.build_eval_scores(
            ChunkScoreInputs(
                pending=pending,
                prev_lens_cpu=prev_lens.numpy(),
                prev_lens_device=prev_lens.to(device),
                chunk_len=chunk_len,
                cache_view=cache_view,
            ),
            geometry,
        )

        # Publish the regime's locked counts, so every reader sees one value.
        self.workspace.locked[req.row].copy_(locked)

        # COUNT is one shared length per cluster, POSITION each member's own
        # top-scored slots; grouping similar-retention heads is what makes the
        # single stored length approximate each member's ideal. The ranking is
        # cached whenever an eval region exists, ``floor_min`` being able to
        # force a positive count even at ratio 0; the COUNT only above zero,
        # where ``compute_counts`` is defined.
        if eval_len > 0 and adjusted_ratio < 1.0:
            req.borrowed_sorted_indices = self._rank_positions(
                eval_scores, num_layers, num_kv_heads, num_groups)
            if adjusted_ratio > 0.0:
                # A count must never reach into a ragged region's padding.
                req.cached_k_new_cpu = np.minimum(
                    self.scope.compute_counts(
                        eval_scores, adjusted_ratio, self.member_to_cluster,
                        num_layers, num_kv_heads, num_groups),
                    geometry.real_eval_len)
            else:
                req.cached_k_new_cpu = None
        else:
            req.borrowed_sorted_indices = None
            req.cached_k_new_cpu = None
        req.locked_count_cpu = locked.cpu().numpy().astype(np.int64)
        req.real_eval_len_cpu = geometry.real_eval_len

        decision = KeepDecision(
            sink_size=int(sink_size),
            tail_size=int(geometry.tail_size),
            adjusted_ratio=float(adjusted_ratio),
            eval_len=int(eval_len),
            budget_tokens=params.budget_tokens,
        )
        req.cross_layer_decision = decision
        req.cached_kept_lengths_cpu = None
        return decision

    def _rank_positions(
        self,
        eval_scores: torch.Tensor,
        num_layers: int,
        num_kv_heads: int,
        num_groups: int,
    ) -> torch.Tensor:
        """Per-member descending POSITION ranking in the executor's
        ``[num_layers, num_groups, page_group_size, width]`` layout, read as
        ``sorted_idx[layer, group, col, :k_aligned]``. Every budget scope shares
        it: the COUNT differs between scopes, the ranking does not.

        Two details keep this allocation-free. Scores are scattered into cluster
        order BEFORE sorting rather than indices after, a score being half the
        width of an int64 index and the target slab already held. And the sort
        covers the slab's FULL width with the unused tail at the dtype minimum,
        so both outputs are contiguous -- a narrower slice is not, and
        ``torch.sort`` would allocate the very temporary this avoids. Padding
        sorts last and the count is clamped to the genuine width, so it can
        never be selected.
        """
        eval_len = eval_scores.shape[-1]
        num_clusters_total = num_layers * num_groups
        width = self.workspace.spec.eval_capacity
        rank_scores = self.workspace.rank_scores.view(
            num_clusters_total, self.page_group_size, width)
        sorted_index = self.workspace.sorted_index.view(
            num_clusters_total, self.page_group_size, width)
        if eval_len < width:
            rank_scores[:, :, eval_len:] = torch.finfo(
                rank_scores.dtype).min
        rank_scores[self.member_to_cluster, self.member_to_col, :eval_len] = (
            eval_scores.reshape(num_layers * num_kv_heads, eval_len))
        torch.sort(rank_scores, dim=-1, descending=True,
                   out=(rank_scores, sorted_index))
        return self.workspace.sorted_index

    def compute_kept_lengths_per_rank(
        self,
        req_id: str,
        eff_seq_lens_row: np.ndarray,
        chunk_len: int,
        floor_min: int,
    ) -> np.ndarray:
        """Compute this chunk's per-(layer, group) post-evict kept_lengths.

        The single source of truth for how many slots each entry keeps. Three
        passes, because a pooling scope shares one total over a span: collect
        each entry's block-rounded demand, enforce the budget (per span when the
        scope pools, else per entry), turn the counts back into lengths.

        Touches no KV cache. ``run_request`` reads the cached result back rather
        than recomputing, so the two agree by construction. Under TP the caller
        may MAX-reduce it across ranks first, to keep the block pool
        consistent."""
        req = self.req_state.get(req_id)
        if req is None or req.cross_layer_decision is None:
            raise RuntimeError(
                f"compute_kept_lengths_per_rank({req_id}): "
                "cross_layer_decision missing — prepare_keep_decision "
                "must run first.")
        keep_dec = req.cross_layer_decision
        sink_size = keep_dec.sink_size
        tail_size = keep_dec.tail_size
        adjusted_ratio = keep_dec.adjusted_ratio
        eval_len = keep_dec.eval_len
        budget_tokens = keep_dec.budget_tokens

        num_layers = self.num_layers
        num_groups = self.num_head_groups_per_layer
        block_size = self.block_size
        floor_min_int = int(floor_min)

        prev_lens = eff_seq_lens_row.astype(
            np.int64, copy=False).reshape(num_layers, num_groups)
        total_seen = prev_lens + chunk_len

        # Keep everything; under a budget, the "cache still fits" path.
        if adjusted_ratio >= 1.0:
            kept_lengths = total_seen.astype(np.int32)
            req.cached_kept_lengths_cpu = kept_lengths
            return kept_lengths

        locked_cpu = req.locked_count_cpu
        k_new_cpu = req.cached_k_new_cpu
        real_eval_len = req.real_eval_len_cpu

        # The workspace's ceiling, not the raw budget: reading back the width
        # a pooling scope was sized for is what keeps the two agreeing.
        pooled_span = (self.scope.pooled_span
                       if budget_tokens is not None else None)
        per_group_capacity = (self.workspace.spec.per_group_capacity
                              if budget_tokens is not None else 0)

        # Pass 1 — per entry: block-rounded demand, floor, dropped remainder.
        want = np.zeros((num_layers, num_groups), dtype=np.int64)
        base = np.zeros((num_layers, num_groups), dtype=np.int64)
        remainder = np.zeros((num_layers, num_groups), dtype=np.int64)
        for layer_idx in range(num_layers):
            for group_idx in range(num_groups):
                total_seen_g = int(total_seen[layer_idx, group_idx])
                locked_count = int(locked_cpu[layer_idx, group_idx])
                eval_len_g = int(real_eval_len[layer_idx, group_idx])
                if eval_len_g <= 0:
                    continue
                # adjusted_ratio == 0 ⇒ no sort cached, keep none.
                k_new = (int(k_new_cpu[layer_idx, group_idx])
                         if k_new_cpu is not None else 0)
                kept_now = (
                    sink_size + locked_count + k_new + tail_size)
                # A floor cannot exceed what the cache holds, nor the budget.
                target_floor = min(floor_min_int, total_seen_g)
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
                budget_tokens - sink_size - locked_cpu - tail_size, 0)
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
                new_locked = (int(locked_cpu[layer_idx, group_idx])
                              + int(keep_counts[layer_idx, group_idx]))
                kept_length = sink_size + new_locked + tail_size
                total_seen_g = int(total_seen[layer_idx, group_idx])
                if kept_length > total_seen_g:
                    kept_length = total_seen_g
                kept_lengths[layer_idx, group_idx] = kept_length
        req.cached_kept_lengths_cpu = kept_lengths
        if self.keep_decision_observer is not None:
            self.keep_decision_observer.record(
                req_id,
                kept_lengths=kept_lengths,
                total_seen=total_seen,
                sink_size=sink_size,
                win_size=tail_size,
                eval_len=eval_len,
            )
        return kept_lengths

    def compact_cluster_stats(
        self,
        req_id: str,
        compressed_layer_idx: int,
        group_idx: int,
        keep_positions: torch.Tensor,
        kept_length: int,
    ) -> None:
        """Follow one (layer, head-group)'s KV eviction in the score memory.

        Called by the executor right after it gathers a cluster's kept KV, with
        the same per-column position matrix, so a score stays slot-aligned with
        its KV and an evicted position's statistics are released. A no-op under
        the ratio regime, whose score memory holds nothing that outlives the
        eviction.

        ``compressed_layer_idx`` is in COMPRESSED layer space -- the space the
        compressor's caches and cluster ids live in -- not the physical layer
        the executor addresses the cache with.
        """
        req = self.req_state.get(req_id)
        if req is None:
            return
        cluster_id = (
            compressed_layer_idx * self.num_head_groups_per_layer + group_idx)
        req.score_store.compact_cluster(
            cluster_id, keep_positions, kept_length)

    def _assert_once_only(
        self,
        req: "_RequestCompressState",
        prev_lens: torch.Tensor,
        num_layers: int,
        num_groups: int,
    ) -> None:
        """``prev_lens`` must equal the kept lengths the last eviction
        committed, or be all-zero before the first one. An interleaved decode
        step would have advanced them."""
        del num_layers, num_groups
        if not req.has_committed:
            if (prev_lens != 0).any():
                bad = int((prev_lens != 0).any(dim=1).long().argmax())
                raise RuntimeError(
                    f"once-only violated: layer {bad} "
                    f"prev_lens={prev_lens[bad].tolist()} but no prior state.")
            return
        valid_cpu = self.workspace.valid_lengths[req.row].cpu()
        if not torch.equal(valid_cpu, prev_lens):
            bad = int((valid_cpu != prev_lens).any(dim=1).long().argmax())
            raise RuntimeError(
                f"once-only violated: layer {bad} "
                f"prev_lens={prev_lens[bad].tolist()} "
                f"valid_lens={valid_cpu[bad].tolist()}.")

    def _take_pending(
        self,
        req: "_RequestCompressState",
        num_layers: int,
        chunk_len: int,
    ) -> torch.Tensor:
        """Consume this chunk's scores as one ``[L, num_kv_heads, chunk_len]``
        view of the row's pending slab, rewinding the per-layer write cursors.

        Every compressible layer must have contributed exactly ``chunk_len``
        tokens — the scorers run in lockstep with the forward pass, so a
        mismatch means a layer's scorer did not fire and the keep decision would
        silently rank stale scores.
        """
        if not self.chunk_scoring_enabled:
            # The scorer never ran by design; an empty view is well-formed.
            return self.workspace.pending_score[req.row, :, :, :0]
        cursors = self.workspace.pending_len[req.row, :num_layers]
        bad = np.flatnonzero(cursors != chunk_len)
        if bad.size:
            layer = int(bad[0])
            raise RuntimeError(
                f"layer {layer}: scored {int(cursors[layer])} tokens for this "
                f"chunk but the chunk is {chunk_len} — receive_score must run "
                "for every compressible layer.")
        self.workspace.pending_len[req.row, :num_layers] = 0
        return self.workspace.pending_score[req.row, :, :, :chunk_len]

    def commit_chunk(
        self,
        req_id: str,
        new_locked: np.ndarray,
        kept_lengths: np.ndarray,
    ) -> None:
        """Record one eviction's result: the positions now permanently kept and
        the length each (layer, group) was cut to. Called once the KV is
        actually rewritten, so the next chunk's once-only check and the
        regime's locked counts read committed state, not intent.
        """
        state = self.req_state.get(req_id)
        if state is None:
            raise RuntimeError(
                f"KVCompressor.commit_chunk: '{req_id}' not begin_request'd.")
        row = state.row
        self.workspace.locked[row].copy_(
            torch.from_numpy(new_locked.astype(np.int64)))
        self.workspace.valid_lengths[row].copy_(
            torch.from_numpy(kept_lengths.astype(np.int64)))
        state.has_committed = True

    def attach_scorers(
        self,
        parents: list[nn.Module],
        inners: list[nn.Module],
    ) -> None:
        """Wire the per-layer scorer into the model (axis 2), compile-safely.

        ``parents[i]`` is layer i's outer block (has hidden_states),
        ``inners[i]`` its inner ``Attention`` (has post-RoPE query/key). No
        forward hooks: torch.compile skips them when it inlines a module
        forward, which would silently drop all scoring. Both kinds are
        delivered through piecewise SPLITTING ops instead -- they must stay
        listed in ``CompilationConfig._attention_ops`` -- so their Python
        bodies run eagerly between captured CUDA-graph pieces every step. A qk
        scorer goes on the inner ``Attention`` as ``compression_qk_scorer``,
        invoked from the ``vllm::unified_attention_ragged`` body; a
        hidden_states scorer as ``compression_gate_capture``, invoked by a
        ``vllm::tangram_gate_capture`` call wrapped in front of the outer
        block's forward, which dynamo does trace.

        Each fires only while ``compress_active`` and ``pending_req_offsets``
        are set, so the dense path pays nothing. The caller owns the ordering:
        ``parents[i]`` / ``inners[i]`` must correspond to ``scorers[i]``."""
        if self.scorer_consumes not in ("hidden_states", "qk"):
            raise ValueError(
                f"attach_scorers: unknown scorer_consumes "
                f"{self.scorer_consumes!r}.")
        if len(inners) != len(self.scorers) or len(parents) != len(inners):
            raise ValueError(
                f"attach_scorers: got {len(inners)} inner layers / "
                f"{len(parents)} parents but {len(self.scorers)} scorers.")
        for layer_idx, (parent, inner, scorer) in enumerate(
                zip(parents, inners, self.scorers)):
            if self.scorer_consumes == "qk":
                # ExpectedAttention needs the outer block's ``rotary_emb``.
                inner.compression_qk_scorer = self._make_qk_scorer(
                    layer_idx, scorer, parent)
            else:
                inner.compression_gate_capture = (
                    self._make_hidden_states_capture(layer_idx, scorer))
                _wrap_forward_with_gate_capture(parent, inner.layer_name)

    def _make_hidden_states_capture(self, layer_idx: int, scorer: nn.Module):
        """Capture fn for hidden_states scorers, invoked by
        ``vllm::tangram_gate_capture`` with the outer block's input
        hidden_states. Concatenates every compression-active request slice into
        one forward per layer to amortise the launch -- per-token gate scores
        are request-independent."""

        def capture(hidden_states: torch.Tensor,
                    _idx=layer_idx, _scorer=scorer) -> None:
            if not self.compress_active or not self.chunk_scoring_enabled:
                return
            offsets = self.pending_req_offsets
            if not offsets:
                return

            valid = [(req, start, end)
                     for req, start, end in offsets if end > start]
            if not valid:
                return

            with torch.no_grad():
                if len(valid) == 1:
                    req, start, end = valid[0]
                    score = _scorer(hidden_states[start:end])
                    self.receive_score(req, _idx, score)
                else:
                    slices = [hidden_states[start:end]
                              for _, start, end in valid]
                    full = torch.cat(slices, dim=0)
                    # [num_kv_heads, sum(end - start)]
                    score_full = _scorer(full)
                    cursor = 0
                    for req, start, end in valid:
                        length = end - start
                        self.receive_score(
                            req, _idx,
                            score_full[:, cursor:cursor + length],
                        )
                        cursor += length

        return capture

    def _make_qk_scorer(
        self, layer_idx: int, scorer: nn.Module, parent: nn.Module,
    ):
        """Scorer fn for qk scorers, invoked from the ragged attention op body
        with its token-major post-RoPE query / key / value. Each request's chunk
        is scored independently: the observation window is chunk-relative, so
        request slices must NOT be concatenated.

        Uniform contract ``scorer(query, key, value, *, module,
        position_offset)``, arguments ignored by scorers that do not need them.
        ``value`` feeds value-norm reweighting. ``module`` is the OUTER block,
        the one owning ``rotary_emb`` -- the inner op only runs the kernel and
        has no RoPE. ``position_offset`` is the chunk's global start
        position."""

        def score_qk(query: torch.Tensor, key: torch.Tensor,
                     value: torch.Tensor | None,
                     _idx=layer_idx, _scorer=scorer, _parent=parent) -> None:
            if not self.compress_active or not self.chunk_scoring_enabled:
                return
            offsets = self.pending_req_offsets
            if not offsets:
                return
            pos_offsets = self.pending_req_pos_offsets or {}

            with torch.no_grad():
                for req, start, end in offsets:
                    if end <= start:
                        continue
                    value_slice = None if value is None else value[start:end]
                    score = _scorer(
                        query[start:end],
                        key[start:end],
                        value_slice,
                        module=_parent,
                        position_offset=pos_offsets.get(req, 0),
                    )
                    self.receive_score(req, _idx, score)

        return score_qk
