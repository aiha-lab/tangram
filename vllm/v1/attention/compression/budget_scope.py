# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Budget scope — compression axis 1.

The aggregation rule that turns per-(layer, head, position) eval scores into a
per-(layer, group) kept COUNT. Every scope shares the same POSITION ranking, so
they differ only in how many positions each entry keeps. ``UniformScope``
thresholds nothing; ``LayerScope`` (default) and ``GlobalScope`` threshold
across a span at cluster granularity and are TP=1 only.

Each scope declares the cluster-map scope it pairs with. That pairing is
required, not a default: the wrong map is silently incorrect.
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

    A scope produces only the COUNT: the POSITION ranking is shared and built
    by the compressor, so a scope never touches paging geometry. Stateless, one
    shared instance per compressor.
    """

    #: The ``compression_budget_scope`` value that selects this scope.
    name: str

    #: Cluster-map scope this pairs with: ``"global"``, ``"per_layer"`` or
    #: ``None``. Declared here because the wrong map is silently incorrect.
    cluster_map_scope: str | None = None

    #: The entries that share ONE budget total: ``"layer"``, ``"global"``, or
    #: ``None`` for no pooling. Owned by the scope, being the same range its
    #: threshold spans. A pooling scope must NOT have its entries capped one by
    #: one -- that forbids the very imbalance the threshold creates.
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

        ``eval_scores`` is ``[num_layers, num_kv_heads, eval_len]``, the
        per-rank head count under TP. Invoked only on the genuine compression
        path (``eval_len > 0`` and ``0 < adjusted_ratio < 1``), so an
        implementation need not handle the fast or zero edges.
        """


class _ClusterCalibratedScope(BudgetScope):
    """Shared machinery for the threshold-based scopes (``global`` / ``layer``).

    Both threshold at CLUSTER granularity: per position the cluster score is
    the MAX over its members, the threshold keeps the top ``adjusted_ratio``
    fraction of (cluster, position) cells, and a cluster's kept COUNT IS its
    physical length -- so the total physical KV is exactly ``adjusted_ratio``
    of the context, sink and block alignment aside, while strong clusters keep
    more. The subclasses differ only in where the threshold spans.

    TP=1 only: member heads are sharded across ranks, so the max-pool would need
    a cross-rank gather first. Config validation rejects TP>1; this is the
    runtime backstop.
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
        flat = eval_scores.reshape(num_layers * num_kv_heads, eval_len)
        # Per-(cluster, position) score = MAX over the cluster's members.
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
        """Kept COUNT ``[num_layers, num_groups]`` from the max-pooled
        ``[num_layers * num_groups, eval_len]`` scores. The count above the
        threshold IS the cluster's physical length."""


class GlobalScope(_ClusterCalibratedScope):
    """One threshold over EVERY layer at once, so strong layers keep more and
    weak ones less.

    The threshold spans all clusters together, so cluster ids need not encode
    their layer: this pairs with a cross-layer map (``--cluster-scope global``).
    Sensitive to cross-layer score-scale disparity -- a scorer giving one layer
    systematically larger scores lets it monopolise the budget."""

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
        # ``topk(n+1).min`` is O(N) vs a full sort: the smallest of the
        # top-(n+1) cells is the cut. Empty clusters stay at -inf.
        flat = cluster_scores.reshape(-1)
        n = max(int(flat.numel() * adjusted_ratio) - 1, 0)
        threshold = torch.topk(flat, k=n + 1).values.min()
        return (
            cluster_scores > threshold
        ).sum(dim=-1).view(num_layers, num_groups)  # [num_layers, num_groups]


class LayerScope(_ClusterCalibratedScope):
    """One threshold PER layer, so no layer can monopolise the budget. Immune
    to cross-layer score-scale disparity.

    Clusters are bucketed by ``cluster_id // num_groups``, which REQUIRES a
    within-layer map (``--cluster-scope per_layer``) for that expression to be
    the physical layer. A cross-layer map numbers clusters by global score
    fill-order with no layer correspondence, so the bucketing would group
    unrelated clusters."""

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
    """Uniform count (the reference's ``pair-head``): every (layer, group)
    keeps the same ``floor(adjusted_ratio * eval_len)``. Only the count is
    uniform -- the shared POSITION ranking still lets each head keep its own
    top-k -- so every entry's ``kept_lengths`` stays identical chunk after
    chunk.

    ``pooled_span`` is ``None`` because with equal counts "each entry within
    ``budget``" and "the span total within ``groups x budget``" are the same
    constraint, and the per-entry form is cheaper."""

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

#: The accepted set in one place; config validation imports it.
BUDGET_SCOPES: tuple[str, ...] = tuple(_SCOPES)

#: Derived from the registry so a new cluster-calibrated scope is covered
#: automatically. Config validation rejects TP>1 with these.
TP1_ONLY_BUDGET_SCOPES: frozenset[str] = frozenset(
    name for name, cls in _SCOPES.items()
    if issubclass(cls, _ClusterCalibratedScope)
)

#: Read by ``WorkspaceSpec``, a pooling scope letting one entry outgrow
#: ``budget``, and by the keep decision to pick the apportionment.
POOLED_BUDGET_SCOPES: frozenset[str] = frozenset(
    name for name, cls in _SCOPES.items() if cls.pooled_span is not None
)

#: Derived from the scope classes; the bundled-map resolver reads it.
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
