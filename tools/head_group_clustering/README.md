# head_group_clustering

Build a **cluster map** for Tangram's head-group paging: an assignment of every
`(layer, head)` to a `(cluster, column)` so that KV heads with *similar*
compression-retention budget share a page group. This removes the max-pool
over-allocation that adjacency-based head groups suffer when retention is
non-uniform.

## Why

A head group allocates `ceil(max_member_kept_length / block_size)` pages, so a
group mixing high- and low-retention heads over-allocates for the small ones. The
budget *ranking* of heads is model-intrinsic (stable across inputs), so a small
pilot profile lets us cluster similar-retention heads together; the group max
then sits close to each member and the waste nearly vanishes. On
Qwen2.5-7B-Instruct-1M this cuts the over-allocation from 42.7% to a 2.9%
residual at the same group size (no attention-kernel performance loss).

## Layout

```
head_group_clustering/
├── clustering.py        # pure: rank_score [L,H] -> ClusterMap (no torch/GPU)
├── validate.py          # pure: Spearman rank stability + boundary sensitivity
├── build_cluster_map.py # CLI: retention profile (.npz) -> cluster map (.npz)
└── tests/test_clustering.py
```

Measurement is **not** done here. The tool reuses a per-(layer, head) retention
profile already produced by `tangram_impl/static_budget_profile/collect.py`
(FastKVZip cross-layer global-threshold, ~50 pilot samples). `clustering.py` is
model-independent so it unit-tests without a GPU.

## Usage

```bash
cd /workspace/vllm-asp
python -m tools.head_group_clustering.build_cluster_map \
    --profile /workspace/tangram_impl/static_budget_profiles/qwen25-7b-1m.npz \
    --base-ratio 0.5 \
    --page-group-size 4 \
    --out /workspace/tangram_impl/static_budget_profiles/qwen25-7b-1m.cluster.npz
```

Key flags:

- `--base-ratio` — which base ratio's retention to cluster on (must exist in the
  profile). Lower ratios tend to have a more stable ranking.
- `--page-group-size` — heads per cluster (must divide `num_layers * num_kv_heads`).
- `--aggregate {stored,mean,median}` — how to reduce per-sample retention into the
  ranking score; `stored` uses the profile's pre-aggregated `ratio_per_head`.
- `--max-heads-per-layer-per-cluster` — optional cap on heads from one layer
  sharing a cluster (relevant to the cross-layer write path; see open question
  Q-B2 in the design tree). Default: unconstrained.

## Output schema (`.npz`)

| key | shape | meaning |
|---|---|---|
| `cluster_of` | `[num_layers, num_kv_heads]` int32 | cluster id of each head |
| `column_of` | `[num_layers, num_kv_heads]` int32 | head's column in its cluster |
| `cluster_members` | `[num_clusters, page_group_size, 2]` int32 | reverse map -> (layer, head) |
| `rank_score` | `[num_layers, num_kv_heads]` float32 | retention score used to cluster |
| `page_group_size` | scalar int32 | heads per cluster |
| `base_ratio` | scalar float32 | base ratio the ranking came from |
| `meta` | json bytes | provenance + over-allocation / rank-stability report |

Invariants (checked in tests): the `(layer, head) <-> (cluster, column)` map is a
bijection; `num_clusters == num_layers * num_kv_heads / page_group_size`; members
from the same layer occupy contiguous columns.

## Validation in `meta`

- `over_allocation` — clustered vs per-head-ideal head-fraction totals and the
  residual waste fraction.
- `rank_stability` — Spearman correlation of each pilot sample's per-cell ranking
  against the aggregate; high mean + not-too-low min means the budget ranking is
  model-intrinsic enough to cluster on (Qwen2.5-7B: mean 0.967, min 0.937).
- `boundary_sensitivity` — score gaps at cluster boundaries; tiny gaps flag
  near-tied heads split across clusters.

## Tests

```bash
cd /workspace/vllm-asp && python -m pytest tools/head_group_clustering/tests/ -q
```
