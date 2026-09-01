"""Budget scope — compression axis 1.

A budget scope is the *aggregation rule* that turns the per-(layer, head,
position) eval scores into a per-(layer, group) kept COUNT — i.e. the scope the
retention budget is balanced over. It is the single place axis 1 branches; the
compressor selects one instance at construction (``make_budget_scope``) and
never inspects it again — so a new scope slots in as a new subclass plus one
registry entry, mirroring the axis-2 scorer factory (``build_qk_scorer``).

All scopes share the POSITION ranking (each head keeps its OWN top-scored
positions — owned by ``KVCompressor._rank_positions``, paging geometry that is
scope-independent). They differ ONLY in how many positions each (layer, group)
keeps:

* ``GlobalScope`` ("global") — a single threshold over every (cluster,
  position) cell of every layer at once: strong clusters in any layer keep
  more, weak ones less. Sensitive to cross-layer score-scale disparity (a layer
  with systematically larger scores can monopolise the budget). Needs a
  cross-layer (global) cluster map; TP=1 only.
* ``LayerScope`` ("layer", default) — a separate threshold per layer, so every
  layer keeps its own top ``adjusted_ratio`` fraction while clusters within a
  layer still diverge. Immune to cross-layer scale disparity. Needs a
  within-layer cluster map; TP=1 only.
* ``UniformScope`` ("uniform") — the degenerate case (no threshold): every
  (layer, group) keeps the same fixed count ``floor(adjusted_ratio *
  eval_len)``; positions still differ per head. No cross-head comparison, so no
  threshold and (under TP) no all-gather.

``global`` and ``layer`` are cluster-calibrated: the threshold is applied at
the cluster (shared-block) granularity directly — per position the cluster
score is the MAX over its member heads — so the kept COUNT per cluster IS its
physical length and the budget is exact (sink/window/block-alignment aside).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)


class BudgetScope(ABC):
    """Axis-1 aggregation rule: eval scores -> per-(layer, group) kept COUNT.

    A scope produces only the COUNT; the POSITION ranking is shared and built
    by the compressor, so a scope never touches paging geometry (page columns,
    ``member_to_col``). Scopes are stateless — one shared instance per
    compressor.
    """

    #: Stable identifier for logging / introspection. Matches the
    #: ``compression_budget_scope`` config value that selects this scope.
    name: str

    #: Cluster-map scope this budget scope pairs with: ``"global"``
    #: (cross-layer), ``"per_layer"``, or ``None`` (uses no cluster map). The
    #: bundled-map resolver reads it to pick the map file; declared here since
    #: pairing the wrong map is silently incorrect.
    cluster_map_scope: str | None = None

    #: Range the retention budget is POOLED over — the set of (layer, group)
    #: entries that share ONE total: ``"layer"`` (the groups of one layer),
    #: ``"global"`` (every group of every layer), or ``None`` (no pooling —
    #: each entry holds its own budget). The scope owns this because it is the
    #: same range its threshold spans; the keep decision reads it to know which
    #: entries to apportion together, and a pooling scope may NOT have its
    #: entries capped at ``budget`` one by one (that would forbid exactly the
    #: imbalance the threshold exists to create).
    pooled_span: str | None = None

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


class _ClusterCalibratedScope(BudgetScope):
    """Shared machinery for the threshold-based scopes (``global`` / ``layer``).

    Both decide the budget at the CLUSTER (shared-block) granularity: per
    position the cluster score is the MAX over its member heads (a position
    matters to the cluster if it matters to ANY member), a threshold over those
    cluster scores keeps the top-``adjusted_ratio`` fraction of (cluster,
    position) cells, and the kept COUNT per cluster IS its physical length. So
    the total physical KV is EXACTLY ``adjusted_ratio`` of the context
    (sink/window/block-alignment aside), while strong clusters keep more than
    weak ones.

    The two scopes differ ONLY in where the threshold spans (every layer at
    once vs one per layer); a subclass implements
    ``_counts_from_cluster_scores`` and this base builds the per-(cluster,
    position) max-pooled scores.

    TP=1 only: under tensor parallelism a cluster's member KV heads are sharded
    across ranks, so the per-position max-pool would need a cross-rank gather
    of the member scores first. Config validation rejects TP>1 with these
    scopes (``uniform`` remains available); this runtime guard is the backstop.
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


class GlobalScope(_ClusterCalibratedScope):
    """One threshold over EVERY layer at once — keep the top ``adjusted_ratio``
    fraction of (cluster, position) cells globally, so strong layers keep more
    and weak ones less.

    Because the threshold spans all clusters together (no per-layer bucketing),
    cluster ids need NOT encode their layer — so this scope pairs with a
    cross-layer (global-scope) cluster map (``--cluster-scope global``), unlike
    ``LayerScope``. Sensitive to cross-layer score-scale disparity: a scorer
    that assigns one layer systematically larger scores lets that layer
    monopolise the budget."""

    name = "global"
    cluster_map_scope = "global"
    pooled_span = "global"

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


class LayerScope(_ClusterCalibratedScope):
    """One threshold PER layer — keep each layer's own top ``adjusted_ratio``
    fraction of (cluster, position) cells, so no layer can monopolise the
    budget (immune to cross-layer score-scale disparity).

    Clusters are grouped for the per-layer threshold by their output layer
    (``cluster_id // num_groups``, matching the ``[num_layers, num_groups]``
    layout); this REQUIRES a within-layer cluster map (``--cluster-scope
    per_layer``) so that ``cluster_id // num_groups`` is the physical layer. A
    cross-layer (global-scope) map assigns cluster ids by global score
    fill-order with no layer correspondence, so the per-layer bucketing would
    group unrelated clusters — pair this scope with a per-layer map."""

    name = "layer"
    cluster_map_scope = "per_layer"
    pooled_span = "layer"

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


class UniformScope(BudgetScope):
    """Uniform count (reference ``pair-head``) — every (layer, group) keeps the
    same ``floor(adjusted_ratio * eval_len)``. The shared POSITION ranking still
    lets each head keep its OWN top-``k`` positions, so only the count is
    uniform. Because the count and the chunk geometry are shared, every
    (layer, group)'s ``kept_lengths`` stays identical chunk after chunk.

    ``pooled_span`` stays ``None``: every entry gets the same count, so
    "each entry within ``budget``" and "the span's total within
    ``groups x budget``" are the same constraint, and the cheaper per-entry
    form is kept."""

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


#: Axis-1 registry: ``compression_budget_scope`` value -> scope class. Adding
#: a scope is one new subclass plus one entry here (no branching elsewhere).
_SCOPES: dict[str, type[BudgetScope]] = {
    UniformScope.name: UniformScope,
    LayerScope.name: LayerScope,
    GlobalScope.name: GlobalScope,
}

#: Valid ``compression_budget_scope`` values, so the accepted set lives in one
#: place (config validation imports this rather than re-listing the names).
BUDGET_SCOPES: tuple[str, ...] = tuple(_SCOPES)

#: Scopes that only support TP=1, derived from the registry so a new
#: cluster-calibrated scope is covered automatically. Their per-cluster
#: max-pool would need a cross-rank gather of sharded member scores (see
#: ``_ClusterCalibratedScope``); config validation rejects the combination.
TP1_ONLY_BUDGET_SCOPES: frozenset[str] = frozenset(
    name for name, cls in _SCOPES.items()
    if issubclass(cls, _ClusterCalibratedScope)
)

#: Scopes that pool the budget over more than one (layer, group). Read at
#: startup by ``WorkspaceSpec`` — a pooled scope lets one entry outgrow
#: ``budget`` (its span's TOTAL is what is held), so the score buffers must be
#: sized for that, and by the keep decision to pick the apportionment.
POOLED_BUDGET_SCOPES: frozenset[str] = frozenset(
    name for name, cls in _SCOPES.items() if cls.pooled_span is not None
)

#: ``compression_budget_scope`` -> cluster-map scope it pairs with, derived
#: from the scope classes (the bundled-map resolver reads this).
CLUSTER_MAP_SCOPE_BY_BUDGET_SCOPE: dict[str, str | None] = {
    name: cls.cluster_map_scope for name, cls in _SCOPES.items()
}


def make_budget_scope(scope: str) -> BudgetScope:
    """Axis-1 dispatch — the ONE place the scope is chosen. ``scope`` is
    ``cache_config.compression_budget_scope``:

    * ``"uniform"`` — fixed per-(layer, group) count (no threshold).
    * ``"layer"`` (default) — per-layer threshold (needs a per-layer cluster
      map; TP=1).
    * ``"global"`` — one cross-layer threshold (needs a global cluster map;
      TP=1)."""
    try:
        return _SCOPES[scope]()
    except KeyError:
        raise ValueError(
            f"make_budget_scope: unknown compression_budget_scope {scope!r}; "
            f"expected one of {BUDGET_SCOPES}.") from None
