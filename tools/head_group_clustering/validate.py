"""Rank-stability validation for head-group clustering.

Clustering only needs the *ranking* of which heads retain more KV budget to be
stable across inputs -- not the absolute retention values. A plain
standard-deviation gate on retention is therefore the wrong instrument; instead
we measure how well each sample's per-cell ranking agrees with the aggregate
ranking via Spearman's rank correlation.

Pure numpy (no scipy) so the tool has no extra dependency.
"""
from __future__ import annotations

import numpy as np


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Assign average ranks to values, resolving ties by the mean rank.

    Equivalent to scipy.stats.rankdata(values, method="average") for a 1D array.
    """
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=np.float64)
    # Average the ranks of tied groups.
    sorted_values = values[order]
    tie_start = 0
    for i in range(1, values.shape[0] + 1):
        if i == values.shape[0] or sorted_values[i] != sorted_values[tie_start]:
            if i - tie_start > 1:
                avg = ranks[order[tie_start:i]].mean()
                ranks[order[tie_start:i]] = avg
            tie_start = i
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two 1D arrays of equal length."""
    if a.shape != b.shape:
        raise ValueError("inputs must have the same shape")
    ra = _rankdata_average(a)
    rb = _rankdata_average(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0.0:
        return 0.0
    return float((ra * rb).sum() / denom)


def rank_stability(per_sample_score: np.ndarray) -> dict:
    """Measure how stable the per-(layer, head) ranking is across pilot samples.

    Inputs:
        per_sample_score: float array [num_samples, num_layers, num_kv_heads];
            retention per sample for a single base ratio.

    Output:
        dict with the mean / min Spearman correlation between each sample's
        flattened cell ranking and the aggregate (mean-over-samples) ranking,
        plus the same for the per-layer rankings (a cluster spanning layers cares
        about the global ranking, but per-layer stability is a useful diagnostic).

    Notes:
        A high mean and a not-too-low min indicate the budget ranking is
        model-intrinsic enough to cluster on. A low min flags samples that
        reorder heads -- candidates for inspection or a lower base ratio.
    """
    if per_sample_score.ndim != 3:
        raise ValueError(
            "per_sample_score must be 3D [num_samples, num_layers, num_kv_heads]")
    num_samples = per_sample_score.shape[0]
    aggregate = per_sample_score.mean(axis=0).ravel()

    global_corrs = np.array([
        _spearman(per_sample_score[s].ravel(), aggregate)
        for s in range(num_samples)
    ])

    layer_aggregate = per_sample_score.mean(axis=0)  # [L, H]
    per_layer_corrs = []
    num_layers = per_sample_score.shape[1]
    for layer in range(num_layers):
        corrs = np.array([
            _spearman(per_sample_score[s, layer], layer_aggregate[layer])
            for s in range(num_samples)
        ])
        per_layer_corrs.append(corrs.mean())
    per_layer_corrs = np.array(per_layer_corrs)

    return {
        "global_spearman_mean": float(global_corrs.mean()),
        "global_spearman_min": float(global_corrs.min()),
        "per_layer_spearman_mean": float(per_layer_corrs.mean()),
        "per_layer_spearman_min": float(per_layer_corrs.min()),
        "num_samples": int(num_samples),
    }


def boundary_sensitivity(rank_score: np.ndarray, page_group_size: int) -> dict:
    """Report how tight the score gaps are at cluster boundaries.

    With plain sorted-chunk clustering, cluster boundaries fall every
    ``page_group_size`` cells in the score-descending order. A small score gap at
    a boundary means two near-tied heads were split into different clusters -- a
    cell whose cluster assignment is fragile to ranking noise. We report the
    minimum and mean boundary gap (normalised by the overall score range).
    """
    flat = np.sort(rank_score.ravel())[::-1]
    score_range = float(flat[0] - flat[-1]) or 1.0
    boundary_gaps = [
        float(flat[i - 1] - flat[i])
        for i in range(page_group_size, flat.shape[0], page_group_size)
    ]
    if not boundary_gaps:
        return {"min_boundary_gap_norm": 0.0, "mean_boundary_gap_norm": 0.0}
    gaps = np.array(boundary_gaps) / score_range
    return {
        "min_boundary_gap_norm": float(gaps.min()),
        "mean_boundary_gap_norm": float(gaps.mean()),
    }
