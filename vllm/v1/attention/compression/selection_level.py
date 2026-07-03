# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selection level — compression axis 1.

A selection level is the *aggregation rule* that turns the per-(layer, head,
position) eval scores into a per-(layer, group) kept COUNT. It is the single
place axis 1 branches; the compressor selects one instance at construction
(``make_selection_level``) and never inspects the level again — so a new level
slots in as a new subclass plus one ``make_selection_level`` entry, mirroring
the axis-2 scorer factory (``build_qk_scorer``).

All levels share the POSITION ranking (each head keeps its OWN top-scored
positions — owned by ``KVCompressor._rank_positions``, paging geometry that is
level-independent). They differ ONLY in how many positions each (layer, group)
keeps, along two ORTHOGONAL axes named directly by the level identifier
``{scope}_{granularity}`` (plus the degenerate ``uniform``):

* THRESHOLD SCOPE — ``crosslayer`` vs ``perlayer``. A ``crosslayer`` level uses a
  single global threshold over every layer at once (strong layers keep more,
  weak ones less, but a layer with systematically larger scores can monopolise
  the budget and starve others). A ``perlayer`` level uses a separate threshold
  per layer, so every layer keeps its own top ``ratio`` fraction (immune to that
  cross-layer score-scale disparity).
* CALIBRATION GRANULARITY — ``head`` vs ``cluster``. A ``head`` level thresholds
  each head's own positions, then a cluster's shared length is the MAX over its
  members; that max-pool inflates the physical budget above the nominal ratio (a
  cluster is as long as its hungriest member). A ``cluster`` level instead
  thresholds at the cluster granularity directly (per-position score = MAX over
  member heads), so the kept COUNT per cluster IS its physical length and the
  budget is EXACTLY the nominal ratio — no inflation.

The four threshold-based levels are the 2x2 of these axes:

* ``CrossLayerHeadLevel`` ("crosslayer_head", reference ``pair``) — cross-layer
  global threshold, head-calibrated (inflating). Tangram's historical default.
* ``PerLayerHeadLevel`` ("perlayer_head", reference AdaKV) — per-layer threshold,
  head-calibrated (inflating). The budget rule the ExpectedAttention reference
  (kvpress ``AdaKVPress``) relies on, and the validated pairing for that scorer.
* ``CrossLayerClusterLevel`` ("crosslayer_cluster") — cross-layer global
  threshold, cluster-calibrated (exact budget with cross-layer block sharing;
  the exact-budget counterpart of ``crosslayer_head``). Pairs with a cross-layer
  (global) cluster map; inherits the cross-layer scale-disparity sensitivity;
  TP=1 only.
* ``PerLayerClusterLevel`` ("perlayer_cluster") — per-layer threshold,
  cluster-calibrated (exact per-layer budget). Pairs with a within-layer cluster
  map; TP=1 only.

* ``UniformLevel`` ("uniform", reference ``score.py:_threshold_head`` /
  ``pair-head``) — the degenerate case (no threshold): every (layer, group)
  keeps the same fixed count ``floor(adjusted_ratio * eval_len)``; positions
  still differ per head. No cross-head comparison, so no threshold and (under TP)
  no all-gather.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
)


class SelectionLevel(ABC):
    """Axis-1 aggregation rule: eval scores -> per-(layer, group) kept COUNT.

    A level produces only the COUNT; the POSITION ranking is shared and built
    by the compressor, so a level never touches paging geometry (page columns,
    ``member_to_col``). Levels are stateless — one shared instance per
    compressor.
    """

    #: Stable identifier for logging / introspection. Matches the
    #: ``compression_level`` config value that selects this level.
    name: str

    #: Cluster-map scope this level pairs with: ``"global"`` (cross-layer),
    #: ``"per_layer"``, or ``None`` (uses no cluster map). The bundled-map
    #: resolver reads it to pick the map file; declared here since pairing the
    #: wrong scope is silently incorrect.
    cluster_map_scope: str | None = None

    @abstractmethod
    def compute_counts(
        self,
        eval_scores: torch.Tensor,
        adjusted_ratio: float,
        member_to_cluster: torch.Tensor,
        num_layers: int,
        num_kv_heads: int,
        num_groups: int,
    ) -> np.ndarray:
        """Return the kept COUNT ``[num_layers, num_groups]`` (int64, on CPU).

        ``eval_scores`` is the ``[num_layers, num_kv_heads, eval_len]`` slice of
        the score workspace (``num_kv_heads`` is the per-rank KV-head count
        under tensor parallelism). ``member_to_cluster[m]`` (member row
        ``m = layer * num_kv_heads + head``) maps a member to its global cluster
        id ``layer * num_groups + group``. Invoked only on the genuine
        compression path (``eval_len > 0`` and ``0 < adjusted_ratio < 1``), so
        implementations need not handle the fast (ratio >= 1) / zero
        (ratio <= 0) edges.
        """


class _HeadLevel(SelectionLevel):
    """Shared machinery for the head-calibrated levels.

    Both head-calibrated levels rank each head's own positions and let a
    cluster's shared length be the MAX over its members (members share the
    cluster's physical KV blocks, so they carry a single length). They differ
    ONLY in how the per-head kept COUNT is calibrated — a subclass implements
    ``_per_head_counts`` and this base does the cluster max-pool. ``UniformLevel``
    does not share this max-pool (every count is already equal), so it subclasses
    ``SelectionLevel`` directly.
    """

    def compute_counts(
        self,
        eval_scores: torch.Tensor,
        adjusted_ratio: float,
        member_to_cluster: torch.Tensor,
        num_layers: int,
        num_kv_heads: int,
        num_groups: int,
    ) -> np.ndarray:
        per_head_k_new = self._per_head_counts(
            eval_scores, adjusted_ratio, num_layers, num_kv_heads)
        return self._max_pool_to_clusters(
            per_head_k_new, member_to_cluster, num_layers, num_groups)

    @abstractmethod
    def _per_head_counts(
        self,
        eval_scores: torch.Tensor,
        adjusted_ratio: float,
        num_layers: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        """Per-head kept COUNT, flattened to ``[num_layers * num_kv_heads]``
        (member-row order ``layer * num_kv_heads + head``), ready for the shared
        cluster max-pool."""

    @staticmethod
    def _max_pool_to_clusters(
        per_head_k_new: torch.Tensor,
        member_to_cluster: torch.Tensor,
        num_layers: int,
        num_groups: int,
    ) -> np.ndarray:
        """Reduce per-head counts to per-cluster lengths by MAX over members.

        Members of a cluster share its physical KV blocks, so the cluster must
        be long enough for its hungriest member; the surplus members keep their
        own (fewer) top positions within that shared length."""
        num_clusters_total = num_layers * num_groups
        cluster_k_new = per_head_k_new.new_zeros(num_clusters_total)
        cluster_k_new.scatter_reduce_(
            0, member_to_cluster, per_head_k_new,
            reduce="amax", include_self=True)
        k_new = cluster_k_new.view(num_layers, num_groups)
        return k_new.cpu().numpy().astype(np.int64)


class CrossLayerHeadLevel(_HeadLevel):
    """Cross-layer global threshold, head-calibrated (reference ``pair``) —
    tangram's historical default. Each KV head counts its own above-threshold
    positions against a SINGLE threshold shared across every layer; a cluster's
    shared length is the MAX of those counts over its members (the members share
    the cluster's physical KV blocks, so they carry a single length)."""

    name = "crosslayer_head"
    cluster_map_scope = "global"

    def _per_head_counts(
        self,
        eval_scores: torch.Tensor,
        adjusted_ratio: float,
        num_layers: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        threshold = self._global_threshold(eval_scores, adjusted_ratio)
        return (
            eval_scores > threshold
        ).sum(dim=-1).reshape(num_layers * num_kv_heads)

    @staticmethod
    def _global_threshold(
        eval_scores: torch.Tensor,
        adjusted_ratio: float,
    ) -> float:
        """Single global threshold over PER-HEAD scores (head-cell calibration):
        keep each head's own top ``adjusted_ratio`` fraction. Calibrating on
        heads rather than clusters makes a head's budget independent of how it is
        grouped, so a low-budget head no longer inherits a high neighbour's
        length. ``topk(n+1).min`` is O(N) vs full sort O(N log N). Under TP,
        all-gather the per-head scores so the threshold spans every rank's KV
        heads (same global semantic as single-process FastKVzip)."""
        tp_world_size = get_tensor_model_parallel_world_size()
        if tp_world_size > 1:
            eval_contiguous = eval_scores.contiguous()
            gathered = [
                torch.empty_like(eval_contiguous)
                for _ in range(tp_world_size)
            ]
            torch.distributed.all_gather(
                gathered, eval_contiguous,
                group=get_tp_group().device_group)
            flat = torch.cat(gathered, dim=1).reshape(-1)
        else:
            flat = eval_scores.reshape(-1)
        n = max(int(flat.numel() * adjusted_ratio) - 1, 0)
        return float(torch.topk(flat, k=n + 1).values.min().item())


class PerLayerHeadLevel(_HeadLevel):
    """Per-layer threshold, head-calibrated (reference AdaKV) — a SEPARATE
    threshold for each layer, so every layer keeps its own top ``adjusted_ratio``
    fraction (a fixed per-layer budget) while heads within a layer still diverge.
    Unlike the cross-layer threshold, no single layer can monopolise the global
    budget, so the selection is immune to cross-layer score-scale disparity — the
    budget rule the ExpectedAttention reference (kvpress ``AdaKVPress``) relies
    on."""

    name = "perlayer_head"
    cluster_map_scope = "per_layer"

    def _per_head_counts(
        self,
        eval_scores: torch.Tensor,
        adjusted_ratio: float,
        num_layers: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        thresholds = self._per_layer_thresholds(
            eval_scores, adjusted_ratio, num_layers)
        return (
            eval_scores > thresholds.view(num_layers, 1, 1)
        ).sum(dim=-1).reshape(num_layers * num_kv_heads)

    @staticmethod
    def _per_layer_thresholds(
        eval_scores: torch.Tensor,
        adjusted_ratio: float,
        num_layers: int,
    ) -> torch.Tensor:
        """One threshold PER layer over that layer's PER-HEAD scores: keep each
        layer's own top ``adjusted_ratio`` fraction. ``topk(n+1).min`` per layer
        is O(N) vs full sort. Under TP the KV heads of a layer are split across
        ranks, so all-gather along the head dimension first — the threshold must
        span every rank's heads of that layer (same global-per-layer semantic as
        a single process). Returns ``[num_layers]`` on the score device."""
        tp_world_size = get_tensor_model_parallel_world_size()
        if tp_world_size > 1:
            eval_contiguous = eval_scores.contiguous()
            gathered = [
                torch.empty_like(eval_contiguous)
                for _ in range(tp_world_size)
            ]
            torch.distributed.all_gather(
                gathered, eval_contiguous,
                group=get_tp_group().device_group)
            # [num_layers, num_kv_heads_total, eval_len].
            layer_scores = torch.cat(gathered, dim=1)
        else:
            layer_scores = eval_scores
        flat = layer_scores.reshape(num_layers, -1)
        n = max(int(flat.shape[1] * adjusted_ratio) - 1, 0)
        # ``topk`` along the per-layer score axis; the smallest of the
        # top-(n+1) values is that layer's threshold.
        return torch.topk(flat, k=n + 1, dim=1).values.min(dim=1).values


class _ClusterLevel(SelectionLevel):
    """Shared machinery for the cluster-calibrated levels.

    The head-calibrated levels above rank each HEAD's positions, then a cluster's
    shared length becomes the MAX over its members; that max-pool inflates the
    physical budget well above the nominal ratio (a cluster is as long as its
    hungriest member). The cluster-calibrated levels instead decide the budget at
    the CLUSTER (shared-block) granularity directly: per position the cluster
    score is the MAX over its member heads (a position matters to the cluster if
    it matters to ANY member), a threshold over those cluster scores keeps the
    top-``adjusted_ratio`` fraction of (cluster, position) cells, and the kept
    COUNT per cluster IS its physical length. So the total physical KV is EXACTLY
    ``adjusted_ratio`` of the context (sink/window/block-alignment aside) — no
    max-pool inflation — while strong clusters still keep more than weak ones.

    The two cluster-calibrated levels differ ONLY in threshold SCOPE (cross-layer
    global vs per-layer); a subclass implements ``_counts_from_cluster_scores``
    and this base builds the per-(cluster, position) max-pooled scores.

    TP=1 only: under tensor parallelism a cluster's member KV heads are sharded
    across ranks, so the per-position max-pool would need a cross-rank gather of
    the member scores first.
    """

    def compute_counts(
        self,
        eval_scores: torch.Tensor,
        adjusted_ratio: float,
        member_to_cluster: torch.Tensor,
        num_layers: int,
        num_kv_heads: int,
        num_groups: int,
    ) -> np.ndarray:
        if get_tensor_model_parallel_world_size() > 1:
            raise NotImplementedError(
                f"{self.name} is implemented for TP=1 only; TP>1 needs a "
                "cross-rank gather of each cluster's member scores before the "
                "per-position max-pool.")
        eval_len = eval_scores.shape[-1]
        num_clusters_total = num_layers * num_groups
        # [num_layers * num_kv_heads, eval_len] in member-row order.
        flat = eval_scores.reshape(num_layers * num_kv_heads, eval_len)
        # Per-(cluster, position) score = MAX over the cluster's member heads.
        cluster_scores = flat.new_full(
            (num_clusters_total, eval_len), float("-inf"))
        idx = member_to_cluster.to(torch.int64).unsqueeze(1).expand(-1, eval_len)
        cluster_scores.scatter_reduce_(
            0, idx, flat, reduce="amax", include_self=True)
        counts = self._counts_from_cluster_scores(
            cluster_scores, adjusted_ratio, num_layers, num_groups)
        return counts.cpu().numpy().astype(np.int64)

    @abstractmethod
    def _counts_from_cluster_scores(
        self,
        cluster_scores: torch.Tensor,
        adjusted_ratio: float,
        num_layers: int,
        num_groups: int,
    ) -> torch.Tensor:
        """Kept COUNT ``[num_layers, num_groups]`` from the per-(cluster,
        position) max-pooled scores ``[num_layers * num_groups, eval_len]``. The
        count above the threshold is the cluster's physical length directly."""


class CrossLayerClusterLevel(_ClusterLevel):
    """Cross-layer global threshold, cluster-calibrated — keep the top
    ``adjusted_ratio`` fraction of (cluster, position) cells over EVERY layer at
    once, the cluster-calibrated analogue of ``CrossLayerHeadLevel``.

    This is the exact-budget counterpart of ``crosslayer_head``: it combines the
    cross-layer block sharing (strong layers keep more, weak ones less) of the
    global threshold with the no-inflation property of cluster-calibrated
    budgeting. Because the threshold spans all clusters together (no per-layer
    bucketing), cluster ids need NOT encode their layer — so this level pairs
    with a cross-layer (global-scope) cluster map (``--cluster-scope global``),
    unlike ``PerLayerClusterLevel``. It inherits the global threshold's
    sensitivity to cross-layer score-scale disparity: a scorer that assigns one
    layer systematically larger scores lets that layer monopolise the budget."""

    name = "crosslayer_cluster"
    cluster_map_scope = "global"

    def _counts_from_cluster_scores(
        self,
        cluster_scores: torch.Tensor,
        adjusted_ratio: float,
        num_layers: int,
        num_groups: int,
    ) -> torch.Tensor:
        # One global threshold over EVERY (cluster, position) cell. ``topk(n+1)
        # .min`` is O(N) vs full sort; the smallest of the top-(n+1) cells is the
        # cut. Empty clusters (no members) stay at -inf and never clear it.
        flat = cluster_scores.reshape(-1)
        n = max(int(flat.numel() * adjusted_ratio) - 1, 0)
        threshold = torch.topk(flat, k=n + 1).values.min()
        return (
            cluster_scores > threshold
        ).sum(dim=-1).view(num_layers, num_groups)  # [num_layers, num_groups]


class PerLayerClusterLevel(_ClusterLevel):
    """Per-layer threshold, cluster-calibrated — keep each layer's own top
    ``adjusted_ratio`` fraction of (cluster, position) cells, so no layer can
    monopolise the budget (immune to cross-layer score-scale disparity), the
    cluster-calibrated analogue of ``PerLayerHeadLevel``.

    Clusters are grouped for the per-layer threshold by their output layer
    (``cluster_id // num_groups``, matching the ``[num_layers, num_groups]``
    layout); this REQUIRES a within-layer cluster map (``--cluster-scope
    per_layer``) so that ``cluster_id // num_groups`` is the physical layer. A
    cross-layer (global-scope) map assigns cluster ids by global score fill-order
    with no layer correspondence, so the per-layer bucketing would group
    unrelated clusters — pair this level with a per-layer map."""

    name = "perlayer_cluster"
    cluster_map_scope = "per_layer"

    def _counts_from_cluster_scores(
        self,
        cluster_scores: torch.Tensor,
        adjusted_ratio: float,
        num_layers: int,
        num_groups: int,
    ) -> torch.Tensor:
        cluster_scores = cluster_scores.view(num_layers, num_groups, -1)
        # One threshold per output layer over its (cluster, position) cells.
        per_layer = cluster_scores.reshape(num_layers, -1)
        n = max(int(per_layer.shape[1] * adjusted_ratio) - 1, 0)
        thresholds = torch.topk(
            per_layer, k=n + 1, dim=1).values.min(dim=1).values  # [num_layers]
        return (
            cluster_scores > thresholds.view(num_layers, 1, 1)
        ).sum(dim=-1)  # [num_layers, num_groups]


class UniformLevel(SelectionLevel):
    """Uniform count (reference ``pair-head``) — every (layer, group) keeps the
    same ``floor(adjusted_ratio * eval_len)``. The shared POSITION ranking still
    lets each head keep its OWN top-``k`` positions, so only the count is
    uniform. Because the count and the chunk geometry are shared, every
    (layer, group)'s ``kept_lengths`` stays identical chunk after chunk."""

    name = "uniform"

    def compute_counts(
        self,
        eval_scores: torch.Tensor,
        adjusted_ratio: float,
        member_to_cluster: torch.Tensor,
        num_layers: int,
        num_kv_heads: int,
        num_groups: int,
    ) -> np.ndarray:
        eval_len = eval_scores.shape[-1]
        k_uniform = int(eval_len * adjusted_ratio)
        return np.full(
            (num_layers, num_groups), k_uniform, dtype=np.int64)


#: Axis-1 registry: ``compression_level`` value -> level class. Adding a level
#: is one new subclass plus one entry here (no level branching elsewhere).
_LEVELS: dict[str, type[SelectionLevel]] = {
    CrossLayerHeadLevel.name: CrossLayerHeadLevel,
    PerLayerHeadLevel.name: PerLayerHeadLevel,
    CrossLayerClusterLevel.name: CrossLayerClusterLevel,
    PerLayerClusterLevel.name: PerLayerClusterLevel,
    UniformLevel.name: UniformLevel,
}

#: Valid ``compression_level`` values, so the accepted set lives in one place
#: (config validation imports this rather than re-listing the names).
SELECTION_LEVELS: tuple[str, ...] = tuple(_LEVELS)

#: Levels that only support TP=1, derived from the registry so a new cluster
#: level is covered automatically. Their per-cluster max-pool would need a
#: cross-rank gather of sharded member scores (see ``_ClusterLevel``).
TP1_ONLY_SELECTION_LEVELS: frozenset[str] = frozenset(
    name for name, cls in _LEVELS.items() if issubclass(cls, _ClusterLevel)
)

#: ``compression_level`` -> cluster-map scope it pairs with, derived from the
#: level classes (the bundled-map resolver reads this).
CLUSTER_MAP_SCOPE_BY_LEVEL: dict[str, str | None] = {
    name: cls.cluster_map_scope for name, cls in _LEVELS.items()
}

#: Head-calibrated level (same scope) each TP=1-only cluster level downgrades to
#: under TP>1; config validation uses this instead of rejecting.
TP_FALLBACK_LEVEL: dict[str, str] = {
    "crosslayer_cluster": "crosslayer_head",
    "perlayer_cluster": "perlayer_head",
}


def make_selection_level(level: str) -> SelectionLevel:
    """Axis-1 dispatch — the ONE place the level is chosen. ``level`` is
    ``cache_config.compression_level``, named ``{scope}_{granularity}``:

    * ``"crosslayer_head"`` — cross-layer global threshold, head-calibrated
      (inflating). Tangram's historical default.
    * ``"perlayer_head"`` — per-layer threshold (AdaKV-style), head-calibrated.
    * ``"crosslayer_cluster"`` — cross-layer global threshold,
      cluster-calibrated (exact budget with cross-layer block sharing; global
      map; TP=1).
    * ``"perlayer_cluster"`` (default) — per-layer threshold, cluster-calibrated
      (exact per-layer budget; needs a per-layer cluster map; TP=1).
    * ``"uniform"`` — fixed per-(layer, group) count (no threshold)."""
    try:
        return _LEVELS[level]()
    except KeyError:
        raise ValueError(
            f"make_selection_level: unknown compression_level {level!r}; "
            f"expected one of {SELECTION_LEVELS}.") from None
