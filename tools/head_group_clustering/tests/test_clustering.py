"""Unit tests for head-group clustering (model-independent, no GPU)."""
from __future__ import annotations

import numpy as np
import pytest

from tools.head_group_clustering.clustering import (
    build_clusters,
    over_allocation_stats,
)
from tools.head_group_clustering.validate import rank_stability


def _assert_bijection(cmap, num_layers, num_kv_heads):
    g = cmap.page_group_size
    total = num_layers * num_kv_heads
    # num_clusters invariant.
    assert cmap.num_clusters == total // g
    # Every (cluster, column) is occupied exactly once by a valid head.
    seen_heads = set()
    seen_cells = set()
    for c in range(cmap.num_clusters):
        for col in range(g):
            layer, head = cmap.cluster_members[c, col]
            assert 0 <= layer < num_layers
            assert 0 <= head < num_kv_heads
            seen_heads.add((int(layer), int(head)))
            seen_cells.add((c, col))
            # Forward map agrees with reverse map.
            assert cmap.cluster_of[layer, head] == c
            assert cmap.column_of[layer, head] == col
    assert len(seen_heads) == total  # every head placed exactly once
    assert len(seen_cells) == total


def test_sorted_chunk_bijection_and_homogeneity():
    rng = np.random.default_rng(0)
    num_layers, num_kv_heads, g = 8, 4, 4
    rank_score = rng.random((num_layers, num_kv_heads))
    cmap = build_clusters(rank_score, g)
    _assert_bijection(cmap, num_layers, num_kv_heads)

    # Sorted-chunk clustering must beat any layer-local grouping on waste, and
    # be at least as good as the per-head ideal lower bound.
    stats = over_allocation_stats(rank_score, cmap)
    assert stats["total_clustered"] >= stats["total_ideal"]
    assert 0.0 <= stats["residual_waste_fraction"] < 0.2


def test_columns_group_same_layer_contiguously():
    # A score field where two heads of the same layer end up in one cluster:
    # their columns must be contiguous (the cross-layer write path relies on it).
    num_layers, num_kv_heads, g = 4, 4, 4
    rank_score = np.zeros((num_layers, num_kv_heads))
    # Make layer 0's heads all top-ranked so they cluster together.
    rank_score[0, :] = [1.0, 0.99, 0.98, 0.97]
    cmap = build_clusters(rank_score, g)
    cluster0 = cmap.cluster_of[0, 0]
    members = cmap.cluster_members[cluster0]
    same_layer_cols = sorted(
        int(col) for col, (lyr, _) in enumerate(members) if lyr == 0)
    # Contiguous == max-min+1 == count.
    assert same_layer_cols == list(
        range(same_layer_cols[0], same_layer_cols[0] + len(same_layer_cols)))


def test_per_layer_cap_constraint():
    num_layers, num_kv_heads, g = 4, 4, 4
    rank_score = np.zeros((num_layers, num_kv_heads))
    rank_score[0, :] = [1.0, 0.99, 0.98, 0.97]  # would otherwise cluster together
    cmap = build_clusters(rank_score, g, max_heads_per_layer_per_cluster=1)
    _assert_bijection(cmap, num_layers, num_kv_heads)
    # No cluster may hold more than one head of any layer.
    for c in range(cmap.num_clusters):
        layers = cmap.cluster_members[c, :, 0]
        _, counts = np.unique(layers, return_counts=True)
        assert counts.max() <= 1


def test_non_divisible_raises():
    rank_score = np.zeros((3, 4))  # 12 heads, g=5 does not divide
    with pytest.raises(ValueError):
        build_clusters(rank_score, 5)


def test_rank_stability_perfect_and_noisy():
    # Identical samples -> perfect rank agreement.
    base = np.random.default_rng(1).random((6, 4))
    per_sample = np.stack([base, base, base], axis=0)
    out = rank_stability(per_sample)
    assert out["global_spearman_mean"] == pytest.approx(1.0)
    # Independent random samples -> agreement well below 1.
    noisy = np.random.default_rng(2).random((10, 6, 4))
    out_noisy = rank_stability(noisy)
    assert out_noisy["global_spearman_mean"] < 0.9
