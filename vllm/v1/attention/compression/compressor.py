# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep-decision logic for KV cache compression.

Owns the per-request score buffers and the keep decision. The per-(layer,
group) kept COUNT is delegated to a pluggable selection level (axis 1; see
selection_level.py) — non-uniform global threshold (default) or uniform count.
KV writes and block-table updates live in the FlashAttention backend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from torch import nn

from vllm.logger import init_logger
from vllm.v1.attention.backends.ragged_layout import (
    identity_member_maps,
    load_cluster_map,
    member_maps_from_cluster_map,
)
from vllm.v1.attention.compression.gate import CompressionGate, load_gates
from vllm.v1.attention.compression.gate_capture import (
    _wrap_forward_with_gate_capture,
)
from vllm.v1.attention.compression.selection_level import (
    SelectionLevel,
    make_selection_level,
)
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
    """Per-chunk sink / window / ratio geometry the executor consumes.

    The eval region is the workspace slice ``[eval_start, eval_end)``;
    sink / locked / window positions outside it are auto-kept. The kept COUNT
    and POSITION live in the per-(layer, group) caches (``cached_k_new_cpu`` /
    ``cached_sorted_indices``), not here — the level-specific threshold is an
    internal of ``SelectionLevel`` and never reaches downstream consumers.
    """
    sink_size: int
    win_size: int
    adjusted_ratio: float
    eval_start: int = 0
    eval_end: int = 0


@dataclass
class _LayerCompressState:
    # [num_kv_heads, win_size]: window-region scores from the previous chunk.
    prior_window_scores: torch.Tensor | None = None
    # [G]: tokens already promoted to "kept" by prior compress steps.
    locked_count_per_group: torch.Tensor | None = None
    # [G]: kept length after the most recent compress
    # (= sink + locked + win, clamped to total_seen).
    valid_lengths_per_group: torch.Tensor | None = None
    # [num_kv_heads, accumulated_len]: gate scores accumulated across this
    # chunk's (possibly budget-sliced) sub-chunks; consumed at the boundary
    # step. Equals one step's score in the common unsliced case.
    pending_score: torch.Tensor | None = None


@dataclass
class _RequestCompressState:
    layer_states: dict[int, _LayerCompressState] = field(default_factory=dict)
    cross_layer_decision: KeepDecision | None = None
    # [L, num_kv_heads, win + chunk]: grow-only workspace
    # laid out as [prior_window | pending_score].
    score_workspace: torch.Tensor | None = None
    workspace_size: int = 0
    # [L, G, page_group_size, eval_len]: each KV head's own descending score
    # ranking, placed at its (cluster, column). Rebuilt each chunk.
    cached_sorted_indices: torch.Tensor | None = None
    cached_k_new_cpu: np.ndarray | None = None           # [L, G]
    locked_count_cpu: np.ndarray | None = None           # [L, G]
    # [L, G] int32: post-evict kept_lengths. Under TP the runner
    # cross-rank MAX-reduces this for block-pool consistency.
    cached_kept_lengths_cpu: np.ndarray | None = None


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
        level: str = "crosslayer_head",
    ) -> None:
        assert num_kv_heads % page_group_size == 0, (
            f"num_kv_heads ({num_kv_heads}) must be divisible by "
            f"page_group_size ({page_group_size}).")

        # Selection level (compression axis 1): the aggregation rule
        # turning eval scores into a per-(layer, group) kept COUNT. ``level`` is
        # ``cache_config.compression_level`` (see selection_level.py). Chosen
        # once here; ``prepare_keep_decision`` calls ``self.level.compute_counts``
        # and never branches on the level again.
        self.level: SelectionLevel = make_selection_level(level)
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

        # member->(cluster, column) maps used by the keep decision. Member row
        # m = layer * num_kv_heads_per_layer + head; member_to_cluster[m] is the
        # global cluster id (flat c = layer * num_head_groups_per_layer + group
        # under the identity map) whose physical KV blocks that head shares, and
        # member_to_col[m] the head's column within that cluster's page.
        # Populated by ``set_cluster_map``; the keep decision requires both.
        self.member_to_cluster: torch.Tensor | None = None
        self.member_to_col: torch.Tensor | None = None

        # Optional observer notified of every finalized per-request keep
        # decision (see ``compute_kept_lengths_per_rank``). ``None`` in
        # production: the keep-decision path then does no extra work. Offline
        # tooling attaches an observer to record the decisions it needs (the
        # head-group clustering retention profiler does this via
        # ``vllm.v1.attention.compression.profiling``); the engine itself stays
        # agnostic to what the observer does with them.
        self.keep_decision_observer: KeepDecisionObserver | None = None

        self.req_state: dict[str, _RequestCompressState] = {}

        # ``pending_req_offsets`` is a list of ``(req_id, start, end)``
        # triples giving each compression-active request's token range in
        # the batch's hidden_states. Tokens outside any triple are skipped.
        self.compress_active: bool = False
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

    def set_qk_scorers(
        self,
        scorer_name: str,
        num_q_per_kv: int,
        snap_window: int,
        snap_kernel: int,
        ea_use_covariance: bool = True,
        ea_use_vnorm: bool = True,
        ea_n_future_positions: int = 512,
        ea_epsilon: float = 1e-2,
    ) -> None:
        """Install a gate-free query/key scorer (SnapKV, KeyDiff, StreamingLLM,
        TOVA, ExpectedAttention): one shared stateless instance per compressible
        layer. Scores come from the model's post-RoPE query/key, so no
        checkpoint is loaded. ``num_q_per_kv`` is the per-rank GQA ratio (model
        q-heads / kv-heads); scorers that ignore it (KeyDiff) simply do not use
        it. Per-scorer hyperparameters (SnapKV ``snap_*``; ExpectedAttention
        ``ea_*``) are forwarded but consumed only by their scorer. The concrete
        scorer is chosen by ``build_qk_scorer`` — the one place the gate-free
        scorer type branches — and ``scorer_consumes`` is read off the module so
        the delivery dispatch in ``attach_scorers`` stays scorer-agnostic."""
        scorer = build_qk_scorer(
            scorer_name,
            num_kv_heads=self.num_kv_heads_per_layer,
            num_q_per_kv=num_q_per_kv,
            head_size=self.head_size,
            snap_window=snap_window,
            snap_kernel=snap_kernel,
            ea_use_covariance=ea_use_covariance,
            ea_use_vnorm=ea_use_vnorm,
            ea_n_future_positions=ea_n_future_positions,
            ea_epsilon=ea_epsilon,
        ).to(device=self.device)
        scorer.eval()
        # The scorer is stateless, so all layers share one instance; the list
        # length matches ``num_layers`` for ``attach_scorers``'s zip.
        self.scorers = [scorer for _ in range(self.num_layers)]
        self.scorer_consumes = scorer.consumes

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
        self.req_state[req_id] = _RequestCompressState()

    def end_request(self, req_id: str) -> None:
        # Idempotent for worker shutdown paths.
        self.req_state.pop(req_id, None)

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
        step that runs the keep decision. Per-token scores are chunk-invariant
        (a token's hidden_states is identical regardless of how prefill was
        sliced — chunked prefill keeps full KV and attention is causal), so the
        accumulated buffer is byte-equivalent to the serial baseline's single
        full-chunk score. ``_collect_layer_tensors`` consumes + clears it only
        at a boundary step. The common unsliced case hits the ``prev is None``
        fast path (plain assign, no concat)."""
        if score.shape[0] != self.num_kv_heads_per_layer:
            raise ValueError(
                f"score head dim {score.shape[0]} != "
                f"num_kv_heads_per_layer {self.num_kv_heads_per_layer}.")
        if req_id not in self.req_state:
            raise RuntimeError(
                f"KVCompressor.receive_score: '{req_id}' not "
                "begin_request'd.")
        layer_state = self.req_state[req_id].layer_states.setdefault(
            layer_idx, _LayerCompressState())
        prev = layer_state.pending_score
        layer_state.pending_score = (
            score if prev is None else torch.cat([prev, score], dim=1))

    def prepare_keep_decision(
        self,
        req_id: str,
        prev_seq_lens_per_layer: torch.Tensor,
        chunk_len: int,
        ratio: float,
        window_size: int,
        n_sink_tokens: int,
        total_prompt_tokens: int,
    ) -> KeepDecision:
        """Run the keep decision for one chunk: compute sink / window / ratio
        geometry, then delegate the per-(layer, group) kept COUNT to the active
        selection level (``self.level``). Caches per-(layer, group) sorted
        indices for the executor. Enforces the once-only invariant:
        ``prev_seq_lens_per_layer`` matches the last
        ``valid_lengths_per_group`` (or is all-zero on the first chunk)."""
        if req_id not in self.req_state:
            raise RuntimeError(
                f"prepare_keep_decision: '{req_id}' not begin_request'd.")
        if not (0.0 < ratio <= 1.0):
            raise ValueError(
                f"prepare_keep_decision: ratio must be in (0, 1], got {ratio}.")

        req = self.req_state[req_id]
        num_layers = self.num_layers
        num_groups = self.num_head_groups_per_layer
        num_kv_heads = self.num_kv_heads_per_layer

        # Sink / window sized by the smallest (layer, group) total length.
        prev_lens = prev_seq_lens_per_layer.to(dtype=torch.long).cpu()
        if prev_lens.shape != (num_layers, num_groups):
            raise ValueError(
                f"prev_seq_lens shape {tuple(prev_lens.shape)} != "
                f"({num_layers}, {num_groups}).")
        min_total = int((prev_lens + chunk_len).min().item())
        sink_size = min(n_sink_tokens, min_total)
        win_size = min(window_size, max(0, min_total - sink_size))

        self._assert_once_only(req, prev_lens, num_layers, num_groups)

        # ``adjusted_ratio`` mirrors baseline FastKVzip
        # (wrapper.py:188-194); we hold ``win_size`` fixed, so the
        # window-shrink branch collapses to zero.
        eff_prompt = max(0, int(total_prompt_tokens) - sink_size)
        if ratio >= 1.0 or eff_prompt <= win_size:
            adjusted_ratio = 1.0
        elif ratio * eff_prompt < win_size:
            adjusted_ratio = 0.0
        else:
            adjusted_ratio = max(0.0, min(1.0,
                (ratio * eff_prompt - win_size) / (eff_prompt - win_size)))

        # First chunk skips the locked [sink, sink+win) prefix; subsequent
        # chunks evaluate the whole fresh chunk.
        is_first_chunk = not any(
            ls.valid_lengths_per_group is not None
            for ls in req.layer_states.values())
        eval_start = win_size + sink_size if is_first_chunk else 0
        eval_end = chunk_len
        eval_len = max(0, eval_end - eval_start)

        # Workspace dtype/device follow the gate output (may stay float32
        # even when self.dtype == bfloat16) for byte-equivalence.
        first_score = next(
            (ls.pending_score for ls in req.layer_states.values()
             if ls.pending_score is not None), None)
        if first_score is None:
            raise RuntimeError(
                f"prepare_keep_decision({req_id}): no pending_score "
                "— receive_score must run first.")
        dtype, device = first_score.dtype, first_score.device
        neg_inf = torch.finfo(dtype).min

        workspace_need = win_size + chunk_len
        if (req.score_workspace is None
                or req.workspace_size < workspace_need
                or req.score_workspace.dtype != dtype
                or req.score_workspace.device != device):
            req.score_workspace = torch.empty(
                num_layers, num_kv_heads, workspace_need,
                dtype=dtype, device=device)
            req.workspace_size = workspace_need
        workspace = req.score_workspace
        workspace.fill_(neg_inf)

        pending, prior, prev_locked = self._collect_layer_tensors(
            req, num_layers, num_groups, num_kv_heads,
            win_size, chunk_len, dtype, device, neg_inf)
        if win_size > 0:
            workspace[:, :, :win_size] = prior
        workspace[:, :, win_size:win_size + chunk_len] = pending

        max_locked = (
            prev_lens.to(device) + chunk_len - sink_size - win_size
        ).clamp_min(0)
        locked = torch.minimum(prev_locked, max_locked)
        for layer_idx in range(num_layers):
            req.layer_states[layer_idx].locked_count_per_group = (
                locked[layer_idx])

        # Save next chunk's prior_window from this chunk's tail.
        if adjusted_ratio < 1.0:
            if win_size > 0 and chunk_len >= win_size:
                new_prior = pending[
                    :, :, chunk_len - win_size:].detach().clone()
                for layer_idx in range(num_layers):
                    req.layer_states[layer_idx].prior_window_scores = (
                        new_prior[layer_idx])
            elif win_size == 0:
                empty = torch.empty(
                    num_kv_heads, 0, dtype=dtype, device=device)
                for layer_idx in range(num_layers):
                    req.layer_states[layer_idx].prior_window_scores = empty
            # win > chunk_len is degenerate; leave prior unchanged.

        # Keep decision = COUNT (per-cluster shared length) + POSITION
        # (per-member ranking). Each KV head (member) keeps its OWN top-scored
        # positions; a cluster then shares ONE kept length, because the
        # cluster's members share the same physical KV blocks (so the length is
        # single). Grouping heads with similar retention budgets via the cluster
        # map makes that shared length approximate each member's ideal — the
        # source of the memory saving. Per-member individual lengths are never
        # stored; only the cluster's one length is. Both outputs need the
        # cluster maps bound by ``set_cluster_map``.
        if self.member_to_cluster is None or self.member_to_col is None:
            raise RuntimeError(
                "prepare_keep_decision: cluster maps are unset — "
                "set_cluster_map must run after construction.")

        # POSITION (ratio-independent rank) is cached whenever there is an eval
        # region to keep from, INCLUDING the zero path (adjusted_ratio == 0):
        # the ratio budget keeps no middle there, but floor_min can still force
        # k_aligned > 0, and run_request indexes this ranking to pick the kept
        # positions (omitting it subscripts None on short prompts). COUNT is
        # cached only on the genuine path — compute_counts is undefined at
        # ratio <= 0, and at ratio 0 the base count is 0 (floor_min supplies the
        # retention against the ranking above).
        if eval_len > 0 and adjusted_ratio < 1.0:
            eval_scores = workspace[:, :, eval_start:eval_end]
            req.cached_sorted_indices = self._rank_positions(
                eval_scores, num_layers, num_kv_heads, num_groups)
            if adjusted_ratio > 0.0:
                req.cached_k_new_cpu = self.level.compute_counts(
                    eval_scores, adjusted_ratio, self.member_to_cluster,
                    num_layers, num_kv_heads, num_groups)
            else:
                req.cached_k_new_cpu = None
        else:
            req.cached_sorted_indices = None
            req.cached_k_new_cpu = None
        req.locked_count_cpu = locked.cpu().numpy().astype(np.int64)

        decision = KeepDecision(
            sink_size=int(sink_size),
            win_size=int(win_size),
            adjusted_ratio=float(adjusted_ratio),
            eval_start=int(eval_start),
            eval_end=int(eval_end),
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
        """Per-member descending POSITION ranking, scattered into the executor's
        ``[num_layers, num_groups, page_group_size, eval_len]`` (cluster,
        column) layout. Shared by the uniform and non-uniform paths — the kept
        COUNT differs between them, the POSITION ranking is the same.

        Each member ranks its OWN scores descending; the executor reads
        ``sorted_idx[layer, group, col, :k_aligned]`` for that column's head.
        Member row ``m = layer * num_kv_heads + head``; ``member_to_cluster[m]``
        / ``member_to_col[m]`` (bound by ``set_cluster_map``) place it.
        """
        num_clusters_total = num_layers * num_groups
        eval_len = eval_scores.shape[-1]
        member_eval = eval_scores.reshape(
            num_layers * num_kv_heads, eval_len)
        _, member_sorted = member_eval.sort(dim=-1, descending=True)
        per_head_sorted = member_sorted.new_zeros(
            num_clusters_total, self.page_group_size, eval_len)
        per_head_sorted[
            self.member_to_cluster, self.member_to_col] = member_sorted
        return per_head_sorted.view(
            num_layers, num_groups, self.page_group_size, eval_len)

    def compute_kept_lengths_per_rank(
        self,
        req_id: str,
        eff_seq_lens_row: np.ndarray,
        chunk_len: int,
        floor_min: int,
    ) -> np.ndarray:
        """Compute this chunk's per-(layer, group) post-evict kept_lengths.

        This is the single source of truth for how many token slots each
        (layer, group) keeps. It touches no KV cache; the result is cached on
        ``req.cached_kept_lengths_cpu`` and :py:meth:`CompressionExecutor.run_request`
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
        win_size = keep_dec.win_size
        adjusted_ratio = keep_dec.adjusted_ratio
        eval_len = max(0, keep_dec.eval_end - keep_dec.eval_start)

        num_layers = self.num_layers
        num_groups = self.num_head_groups_per_layer
        block_size = self.block_size
        floor_min_int = int(floor_min)

        prev_lens = eff_seq_lens_row.astype(
            np.int64, copy=False).reshape(num_layers, num_groups)
        total_seen = prev_lens + chunk_len

        # adjusted_ratio >= 1: keep every position in the eval region.
        if adjusted_ratio >= 1.0:
            kept_lengths = total_seen.astype(np.int32)
            req.cached_kept_lengths_cpu = kept_lengths
            return kept_lengths

        locked_cpu = (
            req.locked_count_cpu
            if req.locked_count_cpu is not None
            else np.zeros((num_layers, num_groups), dtype=np.int64))
        k_new_cpu = req.cached_k_new_cpu

        kept_lengths = np.zeros(
            (num_layers, num_groups), dtype=np.int32)
        for layer_idx in range(num_layers):
            for group_idx in range(num_groups):
                total_seen_g = int(total_seen[layer_idx, group_idx])
                locked_count = int(locked_cpu[layer_idx, group_idx])
                if eval_len > 0:
                    # adjusted_ratio == 0 ⇒ no sort cached, keep none.
                    k_new = (int(k_new_cpu[layer_idx, group_idx])
                             if k_new_cpu is not None else 0)
                    kept_now = (
                        sink_size + locked_count + k_new + win_size)
                    target_floor = min(floor_min_int, total_seen_g)
                    if kept_now < target_floor:
                        extra = min(
                            target_floor - kept_now,
                            eval_len - k_new)
                        if extra > 0:
                            k_new += extra
                    k_aligned = (
                        ((k_new + block_size - 1) // block_size)
                        * block_size)
                    k_aligned = min(k_aligned, eval_len)
                else:
                    k_aligned = 0
                new_locked = locked_count + k_aligned
                kept_length = sink_size + new_locked + win_size
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
                win_size=win_size,
                eval_len=eval_len,
            )
        return kept_lengths

    def _assert_once_only(
        self,
        req: "_RequestCompressState",
        prev_lens: torch.Tensor,
        num_layers: int,
        num_groups: int,
    ) -> None:
        """Compression must run only on chunked-prefill: ``prev_lens`` must
        match the prior ``valid_lengths_per_group`` (or be all-zero on the
        first chunk)."""
        ref = next(
            (ls for ls in req.layer_states.values()
             if ls.valid_lengths_per_group is not None
             or ls.locked_count_per_group is not None), None)

        if ref is None:
            if (prev_lens != 0).any():
                bad = int((prev_lens != 0).any(dim=1).long().argmax())
                raise RuntimeError(
                    f"once-only violated: layer {bad} "
                    f"prev_lens={prev_lens[bad].tolist()} but no prior state.")
            return

        device = (ref.valid_lengths_per_group
                  if ref.valid_lengths_per_group is not None
                  else ref.locked_count_per_group).device
        valid = torch.zeros(
            num_layers, num_groups, dtype=torch.long, device=device)
        for layer_idx in range(num_layers):
            ls = req.layer_states.get(layer_idx)
            if ls is not None and ls.valid_lengths_per_group is not None:
                valid[layer_idx] = ls.valid_lengths_per_group.to(torch.long)
        valid_cpu = valid.cpu()
        if not torch.equal(valid_cpu, prev_lens):
            bad = int((valid_cpu != prev_lens).any(dim=1).long().argmax())
            raise RuntimeError(
                f"once-only violated: layer {bad} "
                f"prev_lens={prev_lens[bad].tolist()} "
                f"valid_lens={valid_cpu[bad].tolist()}.")

    def _collect_layer_tensors(
        self,
        req: "_RequestCompressState",
        num_layers: int,
        num_groups: int,
        num_kv_heads: int,
        win_size: int,
        chunk_len: int,
        dtype: torch.dtype,
        device: torch.device,
        neg_inf: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack per-layer pending / prior / prev_locked into [L, ...]
        tensors. Consumes pending_score on every layer."""
        empty_prior = torch.full(
            (num_kv_heads, win_size), neg_inf, dtype=dtype, device=device)
        zero_locked = torch.zeros(num_groups, dtype=torch.long, device=device)
        pending_list: list[torch.Tensor] = []
        prior_list: list[torch.Tensor] = []
        locked_list: list[torch.Tensor] = []
        for layer_idx in range(num_layers):
            state = req.layer_states.get(layer_idx)
            if state is None or state.pending_score is None:
                raise RuntimeError(
                    f"layer {layer_idx}: no pending_score "
                    "— receive_score must run first.")
            fresh = state.pending_score
            state.pending_score = None
            if fresh.shape[1] != chunk_len:
                raise ValueError(
                    f"layer {layer_idx}: pending_score chunk_len "
                    f"{fresh.shape[1]} != {chunk_len}.")
            if fresh.dtype != dtype or fresh.device != device:
                raise RuntimeError(
                    f"layer {layer_idx}: pending_score dtype/device "
                    f"mismatch (got {fresh.dtype}/{fresh.device}, "
                    f"expected {dtype}/{device}).")
            pending_list.append(fresh)

            prior = state.prior_window_scores
            prior_ok = (
                prior is not None and win_size > 0
                and prior.shape[1] == win_size
                and prior.dtype == dtype
                and prior.device == device)
            prior_list.append(prior if prior_ok else empty_prior)

            locked_list.append(
                state.locked_count_per_group.to(torch.long)
                if state.locked_count_per_group is not None
                else zero_locked)

        pending = torch.stack(pending_list, dim=0)
        prior = (torch.stack(prior_list, dim=0) if win_size > 0
                 else torch.empty(
                     num_layers, num_kv_heads, 0,
                     dtype=dtype, device=device))
        prev_locked = torch.stack(locked_list, dim=0)
        return pending, prior, prev_locked

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
            if not self.compress_active:
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
            if not self.compress_active:
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
