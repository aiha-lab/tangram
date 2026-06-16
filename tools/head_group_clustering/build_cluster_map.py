"""CLI: build a head-group cluster map from a retention profile.

Consumes a per-(layer, head) retention profile produced by ``build_profile.py``
(this package; .npz holding per-head retention for one or more base ratios) and
emits a cluster map: an assignment of every (layer, head) to a (cluster, column)
such that heads with similar retention budget share a cluster.

Measurement is *not* re-run here -- this tool reuses an existing profile. The
clustering itself lives in ``clustering.py`` and is model-independent;
rank-stability checks live in ``validate.py``.

The emitted ``.npz`` holds the integer arrays ``cluster_of`` and ``column_of``
of shape ``[num_layers, num_kv_heads]`` plus the scalar ``page_group_size``,
which is the schema ``load_cluster_map`` in
``vllm/v1/attention/backends/head_grouped_layout.py`` validates and consumes.

Example:
    python -m tools.head_group_clustering.build_cluster_map \
        --profile tools/head_group_clustering/cluster_maps/ea/qwen3-4b-instruct-2507/profile.npz \
        --base-ratio 0.3 \
        --page-group-size 4 \
        --out tools/head_group_clustering/cluster_maps/ea/qwen3-4b-instruct-2507/pg4_r0.3.npz

Prebuilt maps live in ``tools/head_group_clustering/cluster_maps/`` (see its
README for the naming convention and schema).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Absolute imports so the module works both as ``python -m
# tools.head_group_clustering.build_cluster_map`` and via direct execution after
# the package directory is on sys.path. No lazy imports (project style rule).
from tools.head_group_clustering.clustering import build_clusters, over_allocation_stats
from tools.head_group_clustering.validate import boundary_sensitivity, rank_stability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", required=True,
        help="path to the retention profile .npz (from static_budget_profile)")
    parser.add_argument(
        "--out", required=True, help="output cluster-map .npz path")
    parser.add_argument(
        "--base-ratio", type=float, default=0.3,
        help="which base ratio's retention to cluster on (must exist in the "
             "profile's base_ratios); default 0.3 matches the service default "
             "compression ratio (decision 2026-05-28, Q-P1)")
    parser.add_argument(
        "--page-group-size", type=int, default=4,
        help="heads per cluster; must divide num_layers * num_kv_heads")
    parser.add_argument(
        "--aggregate", choices=["stored", "mean", "median"], default="stored",
        help="how to reduce per-sample retention into the ranking score: "
             "'stored' uses the profile's ratio_per_head; 'mean'/'median' "
             "recompute from raw_ratio")
    parser.add_argument(
        "--max-heads-per-layer-per-cluster", type=int, default=None,
        help="optional cap on heads from one layer sharing a cluster (Q-B2); "
             "global scope only")
    parser.add_argument(
        "--cluster-scope", choices=["global", "per_layer"], default="global",
        help="'global' clusters across the whole model (typically cross-layer); "
             "'per_layer' clusters within each layer independently. Use "
             "'per_layer' to match a per-layer-threshold selection level "
             "(perlayer_head, perlayer_cluster), which thresholds within each "
             "layer and so needs same-layer clusters to avoid pooling across "
             "disparate cross-layer score scales; use 'global' for the "
             "cross-layer-threshold levels (crosslayer_head, crosslayer_cluster).")
    return parser.parse_args()


def load_profile(path: Path) -> dict:
    """Load the retention profile and decode its embedded JSON metadata."""
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    required = {"ratio_per_head", "base_ratios"}
    missing = required - keys
    if missing:
        raise ValueError(f"profile {path} missing keys: {sorted(missing)}")
    meta = {}
    if "meta" in keys:
        meta = json.loads(bytes(data["meta"]).decode("utf-8"))
    return {
        "ratio_per_head": data["ratio_per_head"],          # [R, L, H]
        "raw_ratio": data["raw_ratio"] if "raw_ratio" in keys else None,  # [R, S, L, H]
        "base_ratios": np.asarray(data["base_ratios"], dtype=np.float64),
        "meta": meta,
    }


def select_ratio_index(base_ratios: np.ndarray, base_ratio: float) -> int:
    """Find the exact base-ratio index, or fail with the available choices."""
    matches = np.where(np.isclose(base_ratios, base_ratio))[0]
    if matches.size == 0:
        raise ValueError(
            f"base_ratio {base_ratio} not in profile base_ratios "
            f"{base_ratios.tolist()}")
    return int(matches[0])


def compute_rank_score(profile: dict, ratio_idx: int, aggregate: str) -> np.ndarray:
    """Reduce the profile into a per-(layer, head) ranking score [L, H]."""
    if aggregate == "stored":
        return profile["ratio_per_head"][ratio_idx].astype(np.float64)
    if profile["raw_ratio"] is None:
        raise ValueError(
            "profile has no raw_ratio; only --aggregate stored is available")
    per_sample = profile["raw_ratio"][ratio_idx]  # [S, L, H]
    if aggregate == "mean":
        return per_sample.mean(axis=0).astype(np.float64)
    return np.median(per_sample, axis=0).astype(np.float64)


def main() -> int:
    args = parse_args()
    profile_path = Path(args.profile)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = load_profile(profile_path)
    ratio_idx = select_ratio_index(profile["base_ratios"], args.base_ratio)
    rank_score = compute_rank_score(profile, ratio_idx, args.aggregate)
    num_layers, num_kv_heads = rank_score.shape

    cmap = build_clusters(
        rank_score, args.page_group_size,
        max_heads_per_layer_per_cluster=args.max_heads_per_layer_per_cluster,
        cluster_scope=args.cluster_scope)

    waste = over_allocation_stats(rank_score, cmap)
    boundary = boundary_sensitivity(rank_score, args.page_group_size)
    stability = None
    if profile["raw_ratio"] is not None:
        stability = rank_stability(profile["raw_ratio"][ratio_idx])

    meta = {
        "source_profile": str(profile_path),
        "source_model": profile["meta"].get("model_name"),
        "source_gate_sha": profile["meta"].get("fastkvzip_gate_sha"),
        "base_ratio": float(args.base_ratio),
        "aggregate": args.aggregate,
        "page_group_size": int(args.page_group_size),
        "num_layers": int(num_layers),
        # Physical indices of the full-attention layers this map's rows cover
        # (carried through from the profile). For a sliding-window hybrid model
        # the cluster map is authored over these layers only; the engine asserts
        # this list equals the running model's full-attention layers before
        # expanding the map to the physical layout. Absent/None for dense maps.
        "static_layer_ids": profile["meta"].get("static_layer_ids"),
        "num_kv_heads": int(num_kv_heads),
        "num_clusters": int(cmap.num_clusters),
        "max_heads_per_layer_per_cluster": args.max_heads_per_layer_per_cluster,
        "cluster_scope": args.cluster_scope,
        "over_allocation": waste,
        "boundary_sensitivity": boundary,
        "rank_stability": stability,
        "tool_version": "v1",
    }
    meta_bytes = json.dumps(meta, indent=2, sort_keys=False).encode("utf-8")

    np.savez(
        out_path,
        cluster_of=cmap.cluster_of.astype(np.int32),
        column_of=cmap.column_of.astype(np.int32),
        cluster_members=cmap.cluster_members.astype(np.int32),
        rank_score=cmap.rank_score.astype(np.float32),
        page_group_size=np.int32(args.page_group_size),
        base_ratio=np.float32(args.base_ratio),
        meta=np.frombuffer(meta_bytes, dtype=np.uint8),
    )

    print(f"[cluster-map] model={meta['source_model']} "
          f"L={num_layers} H={num_kv_heads} g={args.page_group_size} "
          f"clusters={cmap.num_clusters}")
    print(f"[cluster-map] over-allocation: clustered={waste['total_clustered']:.2f} "
          f"ideal={waste['total_ideal']:.2f} "
          f"residual_waste={waste['residual_waste_fraction']:.1%}")
    if stability is not None:
        print(f"[cluster-map] rank stability (Spearman vs aggregate): "
              f"global mean={stability['global_spearman_mean']:.3f} "
              f"min={stability['global_spearman_min']:.3f}")
    print(f"[cluster-map] boundary gap (norm): "
          f"min={boundary['min_boundary_gap_norm']:.4f} "
          f"mean={boundary['mean_boundary_gap_norm']:.4f}")
    print(f"[cluster-map] wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
