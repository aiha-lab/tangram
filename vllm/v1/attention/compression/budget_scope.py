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

    Two steps. A threshold over the scope's span gives every MEMBER the count
    it would keep on its own, then a cluster's kept COUNT is the MEAN of its
    members' -- the only rule that spends the budget exactly, a page holding no
    padding::

        sum(cluster count x page_group_size) = sum(member count) = the budget

    So the total physical KV is ``adjusted_ratio`` of the context however the
    members are grouped, and what grouping changes is which members are averaged
    together -- a cluster of demanding heads keeps more, and one of quiet heads
    less. The subclasses differ only in where the threshold spans.

    TP=1 only: member heads are sharded across ranks, so the mean would need a
    cross-rank gather first. Config validation rejects TP>1; this is the
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
                "cross-rank gather of each cluster's member counts before the "
                "mean.")
        eval_len = eval_scores.shape[-1]
        flat = eval_scores.reshape(num_layers * num_kv_heads, eval_len)
        member_counts = self._member_counts(
            flat, adjusted_ratio, num_layers, num_kv_heads)
        totals = torch.zeros(
            num_layers * num_groups, dtype=torch.int64, device=flat.device)
        totals.index_add_(
            0, member_to_cluster.to(torch.int64), member_counts)
        # A remainder rounds DOWN, as every budget ceiling here does: rounding
        # up hands the cluster ``page_group_size`` slots the budget does not
        # hold, and there is no rule for whom to take them back from.
        page_group_size = num_kv_heads // num_groups
        counts = torch.div(totals, page_group_size, rounding_mode="floor")
        return counts.view(
            num_layers, num_groups).cpu().numpy().astype(np.int64)

    @abstractmethod
    def _member_counts(
        self,
        flat: torch.Tensor,
        adjusted_ratio: float,
        num_layers: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        """Per-member kept count as int64 ``[num_layers * num_kv_heads]``, from
        the ``[num_layers * num_kv_heads, eval_len]`` scores: how many positions
        this member would keep were it free to choose its own length."""


class GlobalScope(_ClusterCalibratedScope):
    """One threshold over EVERY layer at once, so strong layers keep more and
    weak ones less.

    The threshold spans every layer's members together, so a cluster may draw
    its members from anywhere: this pairs with a cross-layer map
    (``--cluster-scope global``). Sensitive to cross-layer score-scale
    disparity -- a scorer giving one layer systematically larger scores lets it
    monopolise the budget."""

    name = "global"
    cluster_map_scope = "global"
    pooled_span = "global"

    def _member_counts(
        self,
        flat: torch.Tensor,
        adjusted_ratio: float,
        num_layers: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        # ``topk(n+1).min`` is O(N) vs a full sort: the smallest of the
        # top-(n+1) cells is the cut.
        cells = flat.reshape(-1)
        n = max(int(cells.numel() * adjusted_ratio) - 1, 0)
        threshold = torch.topk(cells, k=n + 1).values.min()
        return (flat > threshold).sum(dim=-1)


class LayerScope(_ClusterCalibratedScope):
    """One threshold PER layer, so no layer can monopolise the budget. Immune
    to cross-layer score-scale disparity.

    A member's count comes from its own layer's threshold, so averaging members
    of different layers would mix counts calibrated on different scales. That
    REQUIRES a within-layer map (``--cluster-scope per_layer``); a cross-layer
    map numbers clusters by global score fill-order with no layer
    correspondence, so a cluster would straddle thresholds."""

    name = "layer"
    cluster_map_scope = "per_layer"
    pooled_span = "layer"

    def _member_counts(
        self,
        flat: torch.Tensor,
        adjusted_ratio: float,
        num_layers: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        # One threshold per layer over its own (member, position) cells.
        # ``reshape``, not ``view``: the eval scores are a workspace slice and
        # need not be contiguous.
        per_layer = flat.reshape(num_layers, -1)
        n = max(int(per_layer.shape[1] * adjusted_ratio) - 1, 0)
        thresholds = torch.topk(
            per_layer, k=n + 1, dim=1).values.min(dim=1).values  # [num_layers]
        return (
            flat.reshape(num_layers, num_kv_heads, -1)
            > thresholds.view(num_layers, 1, 1)
        ).sum(dim=-1).reshape(-1)


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
