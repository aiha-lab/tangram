# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep-decision logic for KV cache compression.

Owns the keep decision and delegates every policy choice to one of three
orthogonal, pluggable axes:

* axis 1 — budget scope (``budget_scope.py``): eval scores -> per-(layer,
  group) kept COUNT;
* axis 2 — scorer (``scorer.py`` / ``gate.py``): what a position's score means;
* axis 3 — eviction regime (``eviction_regime.py``): which positions may be
  evicted, how much survives, and how long a score lives.

KV writes and block-table updates live in the FlashAttention backend.
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
    """Contract for an object notified of each finalized per-request keep
    decision. Implemented by offline tooling (not the engine) so the keep
    decision can be observed without entangling the production path with what
    the observer does. ``page_group_size = 1`` makes every ``(layer, group)``
    a single head, so the arrays below are per-(layer, head).

    Args of :meth:`record`:
        req_id: the request whose keep decision this is.
        kept_lengths: ``[num_layers, num_groups]`` int — KV positions retained
            per (layer, group) after eviction (sink + window + selected).
        total_seen: ``[num_layers, num_groups]`` int — full prefill length per
            (layer, group) before eviction.
        sink_size: leading positions kept unconditionally (the sink).
        win_size: trailing recent positions kept unconditionally (the window).
        eval_len: length of the evictable region the threshold ranked over.
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

    The eval region of a (layer, group) is the ``eval_len`` slots starting at
    ``sink_size + locked``; the sink, the locked prefix and the trailing
    ``tail_size`` slots are kept regardless of score. The kept COUNT and
    POSITION live in the per-(layer, group) caches (``cached_k_new_cpu`` /
    ``borrowed_sorted_indices``), not here — the scope-specific threshold is an
    internal of ``BudgetScope`` and never reaches downstream consumers.
    """
    sink_size: int
    #: Trailing always-kept slots: the recent window under the ratio regime,
    #: the whole fresh chunk under the budget regime (unless the fresh chunk is
    #: configured to be evictable).
    tail_size: int
    adjusted_ratio: float
    eval_len: int = 0
    #: Per-(layer, group) hard cap on the post-evict kept length, or ``None``
    #: when the regime sets no absolute cap (the ratio regime).
    budget_tokens: int | None = None


@dataclass
class _RequestCompressState:
    """Per-request bookkeeping. Every tensor lives in the preallocated
    workspace; this holds the row that addresses it plus the small CPU-side
    results the scheduler and the executor read back."""
    #: Workspace row reserved for this request (see ``CompressionWorkspace``).
    row: int
    #: Score memory owned by the active eviction regime (axis 3): a chunk-local
    #: staging view under the ratio regime, a slot-aligned persistent statistics
    #: buffer under the budget regime.
    score_store: RegimeScoreStore
    cross_layer_decision: KeepDecision | None = None
    #: True once an eviction has committed a kept length for this request, so
    #: the next chunk is no longer the first one.
    has_committed: bool = False
    #: [L, G, page_group_size, eval_len] view into the SHARED workspace ranking
    #: buffer: each KV head's own descending score ranking, at its (cluster,
    #: column). Borrowed, not owned — the next request to rank overwrites it, so
    #: it is valid only between this request's ranking and its writeback, which
    #: clears it.
    borrowed_sorted_indices: torch.Tensor | None = None
    cached_k_new_cpu: np.ndarray | None = None           # [L, G]
    locked_count_cpu: np.ndarray | None = None           # [L, G]
    #: [L, G] genuine eval width per (layer, group). Under the budget regime the
    #: eval region is ragged (kept lengths diverge), so the rectangular score
    #: tensor is padded and the selected count must be clamped to this.
    real_eval_len_cpu: np.ndarray | None = None
    #: [L, G] int32 post-evict kept_lengths. Under TP the runner cross-rank
    #: MAX-reduces this for block-pool consistency.
    cached_kept_lengths_cpu: np.ndarray | None = None


def _apportion_blocks(
    want: np.ndarray,
    base: np.ndarray,
    remainder: np.ndarray,
    total: int,
    block_size: int,
) -> np.ndarray:
    """Share one pooled ``total`` over a span's (layer, group) entries.

    A pooled budget scope hands out UNEVEN counts on purpose — that is the
    whole point of ``layer`` / ``global`` — but the selection has to be block
    quantized, and the span's total still has to hold. Capping each entry
    separately does both and destroys the unevenness (a weak entry's unused
    room cannot reach a strong one), so the two are separated here: the block
    quantization is per entry, the budget is per span.

    Largest-remainder apportionment. Every entry first takes the block FLOOR of
    what it asked for, then the blocks that flooring left over go to the
    entries it shortchanged most. Flooring rather than rounding up is what
    makes ``sum <= total`` structural: rounding up first would need the same
    amount taken back afterwards, with no principled rule for whom to take it
    from. Ties break on the lower index, so a rerun of the same step decides
    the same way.

    ``want`` is the most an entry may take (its own block-rounded demand,
    already clamped to its eval width and its physical ceiling); ``base`` its
    block floor; ``remainder`` the demand those floors dropped, which is the
    hand-out order. Example, one layer with two groups, block 16, total 3064:

        want      [2044, 1024]   base [2032, 1008]   remainder [12, 11]
        floors                   -> [2032, 1008] = 3040, one block spare
        spare block to group 0   -> [2048, 1008]  but 2048 > want, skipped
        spare block to group 1   -> [2032, 1024] = 3056
        top-up from the 8 left   -> [2040, 1024] = 3064

    A per-entry cap of 1520 would have returned [1520, 1024] = 2544 instead.
    """
    k = np.minimum(base, want).astype(np.int64)

    # The floors alone can exceed the span's total when ``floor_min`` raised a
    # weak entry's demand upstream. A floor is a request and the budget is a
    # limit, so the budget wins; shave whole blocks off the largest holder
    # first, where there is most to give.
    while int(k.sum()) > total:
        biggest = int(np.argmax(k))
        if k[biggest] <= 0:
            break
        k[biggest] = max(0, int(k[biggest]) - block_size)

    # Hand out the blocks the flooring left over, most-shortchanged entry
    # first, and keep going while any entry can still take another one.
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

    # An entry whose eval region ends mid-block has a ``want`` that is not a
    # block multiple, so the hand-out above can never reach it. Pay that
    # remainder out of what is left, in the same order.
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
    forward pass and read by the per-layer scorers (delivered through the
    attention / gate-capture custom ops; see ``attach_scorers``).
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

        # Budget scope (compression axis 1): the aggregation rule turning
        # eval scores into a per-(layer, group) kept COUNT. ``budget_scope`` is
        # ``cache_config.compression_budget_scope`` (see budget_scope.py).
        # Chosen once here; ``prepare_keep_decision`` calls
        # ``self.scope.compute_counts`` and never branches on the scope again.
        self.scope: BudgetScope = make_budget_scope(budget_scope)
        # The per-entry ceiling the keep decision applies is the width the
        # workspace reserved, and that width depends on the scope: a pooling
        # scope needs room for one entry outgrowing ``budget``. Sized for the
        # wrong scope the decision still runs — it just caps every entry at
        # ``budget`` again and the pooling silently does nothing, which is the
        # defect this pairing exists to prevent.
        if (workspace.spec.slot_capacity > 0
                and workspace.spec.budget_scope != budget_scope):
            raise RuntimeError(
                f"KVCompressor: budget scope '{budget_scope}' does not match "
                f"the workspace, which was sized for "
                f"'{workspace.spec.budget_scope}' "
                f"(per_group_capacity={workspace.spec.per_group_capacity}). "
                "Both come from compression_budget_scope, so they were built "
                "from different configurations.")
        # Eviction regime (compression axis 3): which cached positions may be
        # evicted this chunk, what fraction of them survives, and how long a
        # position's score lives (see eviction_regime.py). Selected by whether
        # ``cache_config.compression_budget_tokens`` is set; like the scope it
        # is chosen once here and never branched on again.
        self.regime: EvictionRegime = make_eviction_regime(regime)
        # Every tensor the keep decision touches lives here, allocated once at
        # startup so the memory-profiling run that follows sizes the KV cache
        # pool around it (see workspace.py).
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

        # Per-layer score producers (axis 2). Populated by
        # ``load_gate_checkpoint`` (FastKVZip; hidden_states) or
        # ``set_qk_scorers`` (gate-free query/key scorers — SnapKV, KeyDiff);
        # kept separate so unit tests can exercise compress() without one.
        # ``scorer_consumes`` ("hidden_states" | "qk") decides how the scorer
        # is delivered: a query/key scorer stored on the inner ``Attention``,
        # or a hidden_states gate wrapped around the outer block (see
        # ``attach_scorers``).
        self.scorers: list[nn.Module] = []
        self.scorer_consumes: str = "hidden_states"
        # Where a live position's score comes from under a fixed budget (see
        # slot_scores.py). Replaced once the scorer is installed, since the
        # scorer is what decides whether the cache can be rescored; the default
        # keeps unit tests that install no scorer working.
        # ``slot_score_source`` is ``cache_config.compression_slot_score_source``
        # — "auto" in production, a source name only when an ablation pins the
        # score provenance while the retention target changes.
        self.slot_score_source_choice = slot_score_source
        self.slot_score_source: SlotScoreSource = PersistedChunkScores()

        # member->(cluster, column) maps used by the keep decision. Member row
        # m = layer * num_kv_heads_per_layer + head; member_to_cluster[m] is the
        # global cluster id (flat c = layer * num_head_groups_per_layer + group
        # under the identity map) whose physical KV blocks that head shares, and
        # member_to_col[m] the head's column within that cluster's page.
        # Populated by ``set_cluster_map``; the keep decision requires both.
        self.member_to_cluster: torch.Tensor | None = None
        self.member_to_col: torch.Tensor | None = None
        # [num_clusters, page_group_size] inverse of the two maps above:
        # (cluster, column) -> member row, or -1 for a column no member fills
        # (a cross-layer map may leave a cluster empty). The eviction regime's
        # score memory needs it because the writeback addresses cluster columns
        # while a score buffer is addressed by member.
        self.cluster_members: torch.Tensor | None = None

        # Optional observer notified of every finalized per-request keep
        # decision (see ``compute_kept_lengths_per_rank``). ``None`` in
        # production: the keep-decision path then does no extra work. Offline
        # tooling attaches an observer to record the decisions it needs (the
        # head-group clustering retention profiler does this via
        # ``vllm.v1.attention.compression.profiling``); the engine itself stays
        # agnostic to what the observer does with them.
        self.keep_decision_observer: KeepDecisionObserver | None = None

        self.req_state: dict[str, _RequestCompressState] = {}

        # Whether the per-chunk scorer has to run at all. A score source that
        # recomputes from the cached keys never reads the chunk's own scores, so
        # the scorer forward and its buffer write are skipped outright.
        self.compress_active: bool = False
        # ``(req_id, start, end)`` token range of each compression-active
        # request in the batch's hidden_states; tokens outside any triple are
        # skipped.
        self.pending_req_offsets: list[tuple[str, int, int]] | None = None
        # ``pending_req_pos_offsets`` maps a compression-active ``req_id`` to
        # the global sequence position of its chunk's first scored token
        # (the request's ``num_computed_tokens`` this step). Query/key scorers
        # that depend on absolute token position (StreamingLLM recency,
        # ExpectedAttention's future-position RoPE rotation) read it in the
        # query/key scorer; position-independent scorers ignore it. Kept
        # parallel to ``pending_req_offsets`` (set/cleared together) so the
        # batch-range tuples stay unchanged and the gate scorer is untouched.
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
        """Install a gate-free query/key scorer (SnapKV, KeyDiff, StreamingLLM,
        TOVA, ExpectedAttention): one shared stateless instance per compressible
        layer. Scores come from the model's post-RoPE query/key, so no
        checkpoint is loaded. ``num_q_per_kv`` is the per-rank GQA ratio (model
        q-heads / kv-heads); scorers that ignore it (KeyDiff) simply do not use
        it. ``options`` carries the settings the selected scorer declared in its
        ``OPTIONS`` (see scorer_options.py), so this signature does not grow when
        a scorer gains a hyperparameter. The concrete scorer comes from
        ``build_qk_scorer`` — a registry lookup, with no scorer branch — and
        ``scorer_consumes`` is read off the module so the delivery dispatch in
        ``attach_scorers`` stays scorer-agnostic."""
        scorer = build_qk_scorer(
            scorer_name,
            num_kv_heads=self.num_kv_heads_per_layer,
            num_q_per_kv=num_q_per_kv,
            head_size=self.head_size,
            options=options,
        ).to(device=self.device)
        scorer.eval()
        # The scorer is stateless, so all layers share one instance; the list
        # length matches ``num_layers`` for ``attach_scorers``'s zip.
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

        ``head_group_cluster_map is None`` selects the identity map (each KV
        head clusters with its adjacent neighbours within a layer); a loaded
        cluster map instead assigns each head to the (possibly cross-layer)
        cluster it shares physical KV blocks with. Either way the maps mirror
        those the FlashAttention builder uses for paging, so scoring and
        physical placement agree (see ``ragged_layout.py``). Member row
        ``m = layer * num_kv_heads_per_layer + head``.
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
        # Invert the two maps once. -1 marks a (cluster, column) no member
        # occupies, so a score buffer never mistakes an empty slot for member 0.
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
        # Idempotent for worker shutdown paths. Releasing the row is what keeps
        # the fixed row pool from leaking across a long-running engine.
        state = self.req_state.pop(req_id, None)
        if state is not None:
            self.workspace.release_row(state.row)

    def receive_score(
        self,
        req_id: str,
        layer_idx: int,
        score: torch.Tensor,
    ) -> None:
        """Stash the scorer-produced ``[num_kv_heads_per_layer, sub_chunk_len]``
        score, accumulating across budget-sliced sub-chunks until the boundary
        step consumes it. Source-agnostic (FastKVZip gate or SnapKV); both feed
        the same buffer.

        Concurrency lets the scheduler split one compression chunk into several
        forward steps (the token budget is shared), and the scorer scores only
        the tokens present in each step. Concatenating in arrival order
        assembles a full ``chunk_size`` of per-token scores by the boundary
        step that runs the keep decision. ``_take_pending`` consumes + rewinds
        it only at a boundary step.

        Byte-equivalence to the serial baseline's single full-chunk score holds
        for the hidden-states gate only: a token's hidden_states does not
        depend on how prefill was sliced (chunked prefill keeps full KV and
        attention is causal), so its score does not either. The qk scorers
        (``snapkv``, the default, and ``tova``) measure a chunk-relative
        observation window — see ``_make_qk_scorer`` — so a sub-chunk's window
        is not the full chunk's, and the assembled buffer differs from the
        serial one. That divergence is accepted, not a defect: the alternative
        is refusing to score until a chunk is whole, which costs the
        concurrency. Do not use bit-identical output as a test oracle for a
        chunk-relative scorer."""
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

        Three delegations, no policy of its own: the active eviction regime
        (axis 3) fixes the geometry — which slots may be evicted, what fraction
        survives — and supplies the eval-region scores from its own score
        memory; the active budget scope (axis 1) turns those scores into a
        per-(layer, group) kept COUNT; and this method caches the per-(layer,
        group) POSITION ranking the executor gathers with.

        Enforces the once-only invariant: ``prev_seq_lens_per_layer`` matches
        the last ``valid_lengths_per_group`` (or is all-zero on the first
        chunk).
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

        # This chunk's scores, as one [L, num_kv_heads, chunk_len] view into the
        # row's pending slab; consuming it rewinds the per-layer write cursors.
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

        # The regime owns the locked counts for this chunk; publish them so the
        # executor and the next chunk read one value.
        self.workspace.locked[req.row].copy_(locked)

        # Keep decision = COUNT (per-cluster shared length) + POSITION
        # (per-member ranking). Each KV head (member) keeps its OWN top-scored
        # positions; a cluster then shares ONE kept length, because the
        # cluster's members share the same physical KV blocks (so the length is
        # single). Grouping heads with similar retention budgets via the cluster
        # map makes that shared length approximate each member's ideal — the
        # source of the memory saving. Per-member individual lengths are never
        # stored; only the cluster's one length is.
        #
        # POSITION (ratio-independent rank) is cached whenever there is an eval
        # region to keep from, INCLUDING the zero path (adjusted_ratio == 0):
        # the ratio budget keeps no middle there, but floor_min can still force
        # k_aligned > 0, and run_request indexes this ranking to pick the kept
        # positions (omitting it subscripts None on short prompts). COUNT is
        # cached only on the genuine path — compute_counts is undefined at
        # ratio <= 0, and at ratio 0 the base count is 0 (floor_min supplies the
        # retention against the ranking above).
        if eval_len > 0 and adjusted_ratio < 1.0:
            req.borrowed_sorted_indices = self._rank_positions(
                eval_scores, num_layers, num_kv_heads, num_groups)
            if adjusted_ratio > 0.0:
                # Clamp to the genuine eval width: under a ragged eval region
                # (the budget regime, where kept lengths diverge) the score
                # tensor is padded, and a count must never reach into padding.
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
        """Per-member descending POSITION ranking, in the executor's
        ``[num_layers, num_groups, page_group_size, width]`` (cluster, column)
        layout. Shared by every budget scope — the kept COUNT differs between
        them, the POSITION ranking is the same.

        Each member ranks its OWN scores descending; the executor reads
        ``sorted_idx[layer, group, col, :k_aligned]`` for that column's head.
        Member row ``m = layer * num_kv_heads + head``; ``member_to_cluster[m]``
        / ``member_to_col[m]`` (bound by ``set_cluster_map``) place it.

        Two details are there to keep this allocation-free. The scores are
        scattered into cluster order BEFORE sorting rather than the indices
        after, because a score is half the width of an int64 index and the
        scatter target is a slab we already hold. And the sort runs over the
        slab's FULL reserved width with the unused tail held at the dtype
        minimum, so both sort outputs are contiguous slabs: a narrower slice
        would be non-contiguous, and ``torch.sort`` would fall back to a
        temporary of exactly the size we are trying not to allocate. Padding can
        never be selected — it sorts last, and the kept count is clamped to each
        (layer, group)'s genuine eval width.
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

        This is the single source of truth for how many token slots each
        (layer, group) keeps. It runs in three passes because a pooling budget
        scope shares one total over a span of entries: pass 1 collects each
        entry's block-rounded demand, pass 2 enforces the budget (per span
        under ``layer`` / ``global``, per entry under ``uniform``), pass 3
        turns the counts back into lengths.

        It touches no KV cache; the result is cached on
        ``req.cached_kept_lengths_cpu`` and
        :py:meth:`CompressionExecutor.run_request`
        reads it back (deriving its top-k span from it via
        ``_new_region_from_kept_length``) rather than recomputing — so the two
        stay consistent by construction. Under TP the caller may cross-rank
        MAX-reduce the cache before ``run_request`` to keep the block pool
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

        # adjusted_ratio >= 1: keep every position in the eval region. Under the
        # budget regime this is the "cache still fits the budget" path.
        if adjusted_ratio >= 1.0:
            kept_lengths = total_seen.astype(np.int32)
            req.cached_kept_lengths_cpu = kept_lengths
            return kept_lengths

        locked_cpu = req.locked_count_cpu
        k_new_cpu = req.cached_k_new_cpu
        # Genuine (unpadded) eval width per (layer, group). Uniform under the
        # ratio regime; ragged under the budget regime.
        real_eval_len = req.real_eval_len_cpu

        # Under a budget the ceiling is the workspace's, not the raw budget:
        # a pooling scope was sized for one entry outgrowing ``budget`` (see
        # ``WorkspaceSpec.per_group_capacity``), and reading it back here is
        # what keeps the reserved width and the decision from disagreeing.
        # ``pooled_span`` is the scope's own declaration of which entries share
        # one total; ``None`` (``uniform``, and every scope under the ratio
        # regime, which has no absolute total to share) keeps the per-entry
        # cap.
        pooled_span = (self.scope.pooled_span
                       if budget_tokens is not None else None)
        per_group_capacity = (self.workspace.spec.per_group_capacity
                              if budget_tokens is not None else 0)

        # Pass 1 — per entry: how much it asks for (``want``, already block
        # rounded and clamped), its block floor, and the demand that flooring
        # dropped. Nothing is shared yet.
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
                # The floor cannot ask for more than the cache holds, nor
                # (under a budget) for more than the budget allows.
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
                    # Rounding the selection UP to a block is what keeps the
                    # kept span page-contiguous, so the entry's own ceiling is
                    # rounded DOWN to a block multiple rather than cutting
                    # mid-block: the kept length then never exceeds the
                    # capacity and stays block-aligned. Under ``uniform`` the
                    # capacity IS the budget, so this is the whole constraint;
                    # under a pooling scope it is only the physical ceiling and
                    # the budget itself is enforced per span in pass 2.
                    room_g = (per_group_capacity - sink_size - locked_count
                              - tail_size)
                    k_aligned = min(
                        k_aligned, max(0, (room_g // block_size) * block_size))
                want[layer_idx, group_idx] = k_aligned
                base[layer_idx, group_idx] = min(
                    (k_new // block_size) * block_size, k_aligned)
                remainder[layer_idx, group_idx] = (
                    k_new - base[layer_idx, group_idx])

        # Pass 2 — per span: a pooling scope's entries share one total, so
        # the budget is enforced over the span instead of entry by entry. The
        # total is the sum of what the BUDGET (not the capacity) leaves each
        # entry, which is where the pooling is: an entry that wants less than
        # its share leaves the rest for the others.
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
        # Emit the finalized decision to an optional observer (None in
        # production — no extra work). Offline tooling uses this to record
        # per-(layer, head) retention without touching the production path.
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
        the same per-column position matrix, so the active regime's score memory
        stays slot-aligned with the KV and an evicted position's statistics are
        released. A no-op under the ratio regime, whose score memory is
        chunk-local and holds nothing that outlives the eviction.

        ``compressed_layer_idx`` is the index in COMPRESSED layer space (the
        space the compressor's caches and the cluster ids live in), not the
        physical layer the executor addresses the KV cache with.
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
        """Compression must run only on chunked-prefill: ``prev_lens`` must
        match the kept lengths the last eviction committed (or be all-zero
        before the first one)."""
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
            # The scorer never ran, by design; hand back an empty view so the
            # score source sees a well-formed (zero-width) chunk.
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
        """Record the result of one eviction: the positions now permanently kept
        and the length each (layer, group) was cut to.

        Called by the executor once the KV has actually been rewritten, so the
        next chunk's ``prev_seq_lens`` check and the regime's locked counts read
        the committed state rather than the intent.
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

        ``parents[i]`` is layer i's outer attention block (exposes
        hidden_states); ``inners[i]`` is its inner ``Attention`` (exposes
        post-RoPE query/key). Neither uses module forward hooks: torch.compile
        skips hooks when it inlines module forwards, which would silently drop
        all scoring under compilation. Instead, both scorer kinds are
        delivered through custom ops that are piecewise *splitting ops*
        (listed in ``CompilationConfig._attention_ops``), so their Python
        bodies execute eagerly between captured CUDA-graph pieces on every
        step:

        - query/key scorers (SnapKV, KeyDiff, ...) are stored on the inner
          ``Attention`` as ``compression_qk_scorer`` and invoked at the top of
          the ``vllm::unified_attention_ragged`` op body, which receives
          the same token-major query/key/value the old pre-hook saw.
        - hidden_states scorers (FastKVZip gate) are stored on the inner
          ``Attention`` as ``compression_gate_capture`` and invoked by a
          ``vllm::tangram_gate_capture`` op call inserted in front of the
          outer block's forward (instance-level wrap; dynamo traces the
          wrapper and keeps the op as an opaque graph node).

        Every delivered fn fires only when ``compress_active`` is True and
        ``pending_req_offsets`` is non-empty (else zero overhead). Caller owns
        the ordering: ``parents[i]`` / ``inners[i]`` must correspond to
        ``scorers[i]``."""
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
                # ``parent`` is threaded through because ExpectedAttention
                # needs the outer block's ``rotary_emb`` (the inner op only
                # runs the attention kernel and has no RoPE).
                inner.compression_qk_scorer = self._make_qk_scorer(
                    layer_idx, scorer, parent)
            else:
                inner.compression_gate_capture = (
                    self._make_hidden_states_capture(layer_idx, scorer))
                _wrap_forward_with_gate_capture(parent, inner.layer_name)

    def _make_hidden_states_capture(self, layer_idx: int, scorer: nn.Module):
        """Capture fn for hidden_states scorers (FastKVZip gate), invoked by
        the ``vllm::tangram_gate_capture`` op with the outer block's input
        hidden_states. Concatenates all compression-active request slices into
        a single forward per layer to amortise kernel launch (per-token scores
        are request-independent)."""

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
        """Scorer fn for query/key scorers (SnapKV, KeyDiff, StreamingLLM,
        TOVA, ExpectedAttention), invoked from the ragged attention op
        body with the op's token-major query / key / value (post-RoPE — the
        same tensors the inner ``Attention``'s forward receives). Scores each
        request's chunk independently — the observation window is
        chunk-relative, so request slices must NOT be concatenated.

        Every qk scorer is called with the uniform contract
        ``scorer(query, key, value, *, module, position_offset)``: ``value``
        feeds value-norm reweighting (ExpectedAttention); ``module`` is the
        OUTER attention block (``parent``), the one that owns ``rotary_emb`` —
        the inner op only runs the attention kernel and has no RoPE — so
        ExpectedAttention can recover pre-RoPE queries / build the future-
        position rotation; ``position_offset`` is the chunk's global start
        position (StreamingLLM recency, ExpectedAttention future positions).
        Scorers that do not need an argument simply ignore it."""

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
