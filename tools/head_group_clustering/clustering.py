"""Pure clustering logic for head-group clustering.

Given a per-(layer, head) ranking score (how much KV-cache budget each head
retains under non-uniform compression), partition all heads of the model into
clusters of ``page_group_size`` heads such that heads with *similar* retention
land in the same cluster. A cluster becomes the unit of page allocation; because
its members retain similar amounts, the group's max-pooled budget stays close to
each member and the per-group over-allocation ("max-pool waste") nearly vanishes.

This module is model-independent and free of GPU / torch dependencies so it can
be unit-tested in isolation. The runtime wiring and the retention measurement
live elsewhere (see ``build_cluster_map.py`` in this package).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusterMap:
    """A complete (layer, head) -> (cluster, column) assignment.

    Attributes:
        cluster_of:      int array [num_layers, num_kv_heads]; the cluster id a
                         head belongs to.
        column_of:       int array [num_layers, num_kv_heads]; the head's column
                         position (0 .. page_group_size-1) inside its cluster's
                         page. Members originating from the same layer occupy
                         contiguous columns, so one layer's contribution to a
                         cluster is a single (column_offset, length) span.
        cluster_members: int array [num_clusters, page_group_size, 2]; the
                         reverse map. cluster_members[c, col] = (layer, head)
                         occupying column ``col`` of cluster ``c``.
        rank_score:      float array [num_layers, num_kv_heads]; the retention
                         score used to cluster, kept for provenance / inspection.
        page_group_size: heads per cluster.
    """

    cluster_of: np.ndarray
    column_of: np.ndarray
    cluster_members: np.ndarray
    rank_score: np.ndarray
    page_group_size: int

    @property
    def num_clusters(self) -> int:
        return int(self.cluster_members.shape[0])


def build_clusters(
    rank_score: np.ndarray,
    page_group_size: int,
    max_heads_per_layer_per_cluster: int | None = None,
    cluster_scope: str = "global",
) -> ClusterMap:
    """Partition all (layer, head) cells into clusters of similar retention.

    Two scopes control which heads may share a cluster (== share physical KV
    blocks and one max-pooled retention budget):

    * ``"global"`` (default): cluster across the whole model. Every (layer, head)
      cell is sorted by ``rank_score`` descending and the sorted order is chunked
      into groups of ``page_group_size`` -- the simplest scheme that drives the
      residual over-allocation close to the per-head ideal.
      Clusters are typically *cross-layer*. ``max_heads_per_layer_per_cluster``
      optionally caps how many heads of one layer may share a cluster.
    * ``"per_layer"``: cluster *within each layer independently*. A layer's heads
      are sorted by score and chunked into ``num_kv_heads // page_group_size``
      clusters, so every cluster's members come from one layer. Use this when the
      selection level thresholds per layer (``perlayer_head`` or
      ``perlayer_cluster``): a cross-layer cluster would force the keep decision
      to pool scores/lengths across layers whose score scales differ by orders
      of magnitude, reintroducing the very disparity per-layer thresholding
      removes. Cluster
      ids are laid out as ``layer * groups_per_layer + group`` so they coincide
      with the layer's own physical block-table rows.

    Inputs:
        rank_score: [num_layers, num_kv_heads] float; higher == retains more KV.
        page_group_size: heads per cluster. Must divide num_layers * num_kv_heads
            (``"global"``) or num_kv_heads (``"per_layer"``).
        max_heads_per_layer_per_cluster: optional cap on how many heads from a
            single layer may share one cluster. ``"global"`` only; None ==
            unconstrained.
        cluster_scope: ``"global"`` or ``"per_layer"`` (see above).

    Output:
        ClusterMap with a bijective (layer, head) <-> (cluster, column) mapping.

    Invariants:
        - Every (layer, head) is assigned to exactly one (cluster, column).
        - num_clusters == num_layers * num_kv_heads / page_group_size.
        - Within a cluster, columns are ordered by (layer, head) ascending.
    """
    if rank_score.ndim != 2:
        raise ValueError(
            f"rank_score must be 2D [num_layers, num_kv_heads], got shape "
            f"{rank_score.shape}")
    if cluster_scope not in ("global", "per_layer"):
        raise ValueError(
            f"cluster_scope must be 'global' or 'per_layer', got "
            f"{cluster_scope!r}")
    num_layers, num_kv_heads = rank_score.shape
    total_heads = num_layers * num_kv_heads
    if page_group_size <= 0:
        raise ValueError(f"page_group_size must be positive, got {page_group_size}")
    if total_heads % page_group_size != 0:
        raise ValueError(
            f"page_group_size ({page_group_size}) must divide total heads "
            f"({num_layers} x {num_kv_heads} = {total_heads})")
    num_clusters = total_heads // page_group_size

    if cluster_scope == "per_layer":
        if max_heads_per_layer_per_cluster is not None:
            raise ValueError(
                "max_heads_per_layer_per_cluster is incompatible with "
                "cluster_scope='per_layer' (every cluster is single-layer by "
                "construction)")
        cluster_of = _assign_per_layer(rank_score, page_group_size)
    else:
        if max_heads_per_layer_per_cluster is not None:
            if max_heads_per_layer_per_cluster <= 0:
                raise ValueError(
                    "max_heads_per_layer_per_cluster must be positive")
            # Feasibility: each layer has num_kv_heads heads to place across
            # clusters; capacity per layer = num_clusters * cap must cover them.
            if num_clusters * max_heads_per_layer_per_cluster < num_kv_heads:
                raise ValueError(
                    "max_heads_per_layer_per_cluster too small to place all "
                    "heads of a layer across the available clusters")
        cluster_of = _assign_global(
            rank_score, page_group_size, num_clusters,
            max_heads_per_layer_per_cluster)

    # Within each cluster, order members by (layer, head) ascending and assign
    # columns, so a single layer's members form a contiguous column span.
    column_of = np.empty((num_layers, num_kv_heads), dtype=np.int64)
    cluster_members = np.empty((num_clusters, page_group_size, 2), dtype=np.int64)
    for c in range(num_clusters):
        member_layers, member_heads = np.where(cluster_of == c)
        if member_layers.size != page_group_size:
            raise AssertionError(
                f"cluster {c} has {member_layers.size} members, expected "
                f"{page_group_size}")
        member_order = np.lexsort((member_heads, member_layers))
        for col, member_idx in enumerate(member_order):
            layer = int(member_layers[member_idx])
            head = int(member_heads[member_idx])
            column_of[layer, head] = col
            cluster_members[c, col] = (layer, head)

    return ClusterMap(
        cluster_of=cluster_of,
        column_of=column_of,
        cluster_members=cluster_members,
        rank_score=rank_score.astype(np.float64),
        page_group_size=page_group_size,
    )


def _pick_cluster(
    cluster_fill: np.ndarray,
    layer_count: np.ndarray,
    layer: int,
    page_group_size: int,
    max_heads_per_layer_per_cluster: int | None,
    num_clusters: int,
) -> int:
    """Return the cluster id to place the next (sorted) cell into.

    Unconstrained: the first cluster with a free column (== plain sequential
    chunking of the sorted order, since clusters fill left to right).
    Constrained: the first cluster that also respects the per-layer cap; if none
    of the partially-filled clusters qualify, the lowest-id empty cluster is used.
    """
    has_room = cluster_fill < page_group_size
    if max_heads_per_layer_per_cluster is None:
        candidates = np.where(has_room)[0]
        return int(candidates[0])

    layer_ok = layer_count[:, layer] < max_heads_per_layer_per_cluster
    candidates = np.where(has_room & layer_ok)[0]
    if candidates.size == 0:
        raise AssertionError(
            "no cluster can accept the cell under the per-layer cap; this should "
            "be prevented by the feasibility check in build_clusters")
    return int(candidates[0])


def _assign_global(
    rank_score: np.ndarray,
    page_group_size: int,
    num_clusters: int,
    max_heads_per_layer_per_cluster: int | None,
) -> np.ndarray:
    """Cross-model cluster assignment: sort all cells by score, chunk by group.

    Returns ``cluster_of`` [num_layers, num_kv_heads]. Clusters are typically
    cross-layer (a group of ``page_group_size`` cells adjacent in global score
    order can come from any layers).
    """
    num_layers, num_kv_heads = rank_score.shape
    total_heads = num_layers * num_kv_heads

    # Flatten and sort cells by score descending. Ties broken by (layer, head)
    # so the assignment is deterministic and reproducible.
    layers, heads = np.meshgrid(
        np.arange(num_layers), np.arange(num_kv_heads), indexing="ij")
    cells = np.stack(
        [rank_score.ravel(), layers.ravel(), heads.ravel()], axis=1)
    order = np.lexsort((cells[:, 2], cells[:, 1], -cells[:, 0]))
    sorted_layers = cells[order, 1].astype(np.int64)
    sorted_heads = cells[order, 2].astype(np.int64)

    assigned_cluster = np.full(total_heads, -1, dtype=np.int64)
    cluster_fill = np.zeros(num_clusters, dtype=np.int64)
    layer_count = np.zeros((num_clusters, num_layers), dtype=np.int64)
    for sorted_idx in range(total_heads):
        layer = int(sorted_layers[sorted_idx])
        target = _pick_cluster(
            cluster_fill, layer_count, layer, page_group_size,
            max_heads_per_layer_per_cluster, num_clusters)
        assigned_cluster[sorted_idx] = target
        cluster_fill[target] += 1
        layer_count[target, layer] += 1

    cluster_of = np.empty((num_layers, num_kv_heads), dtype=np.int64)
    for sorted_idx in range(total_heads):
        cluster_of[sorted_layers[sorted_idx], sorted_heads[sorted_idx]] = (
            assigned_cluster[sorted_idx])
    return cluster_of


def _assign_per_layer(
    rank_score: np.ndarray,
    page_group_size: int,
) -> np.ndarray:
    """Within-layer cluster assignment: cluster each layer independently.

    A layer's ``num_kv_heads`` heads are sorted by score descending (ties broken
    by head id) and chunked into ``num_kv_heads // page_group_size`` clusters, so
    every cluster's members come from one layer. Cluster ids are offset by
    ``layer * groups_per_layer`` so they are globally unique and land in the
    layer's own physical block-table rows (``layer * groups_per_layer + group``).
    Returns ``cluster_of`` [num_layers, num_kv_heads].
    """
    num_layers, num_kv_heads = rank_score.shape
    if num_kv_heads % page_group_size != 0:
        raise ValueError(
            f"cluster_scope='per_layer' needs page_group_size "
            f"({page_group_size}) to divide num_kv_heads ({num_kv_heads})")
    groups_per_layer = num_kv_heads // page_group_size

    cluster_of = np.empty((num_layers, num_kv_heads), dtype=np.int64)
    head_ids = np.arange(num_kv_heads)
    for layer in range(num_layers):
        # Sort this layer's heads by score desc; ties broken by head id so the
        # assignment is deterministic.
        head_order = np.lexsort((head_ids, -rank_score[layer]))
        base = layer * groups_per_layer
        for pos, head in enumerate(head_order):
            cluster_of[layer, int(head)] = base + pos // page_group_size
    return cluster_of


def over_allocation_stats(rank_score: np.ndarray, cmap: ClusterMap) -> dict:
    """Quantify max-pool over-allocation of a cluster map vs the per-head ideal.

    All quantities are in "head-fraction" units: rank_score is a retention ratio
    in [0, 1], so summing it over heads gives the ideal (waste-free) allocation,
    and summing each cluster's max over its members times page_group_size gives
    the actual paged allocation.

    Returns total_clustered, total_ideal, and the residual waste fraction.
    """
    g = cmap.page_group_size
    ideal = float(rank_score.sum())
    clustered = 0.0
    for c in range(cmap.num_clusters):
        members = cmap.cluster_members[c]  # [g, 2]
        member_scores = rank_score[members[:, 0], members[:, 1]]
        clustered += float(member_scores.max()) * g
    residual_waste = (clustered - ideal) / clustered if clustered > 0 else 0.0
    return {
        "total_clustered": clustered,
        "total_ideal": ideal,
        "residual_waste_fraction": residual_waste,
    }
