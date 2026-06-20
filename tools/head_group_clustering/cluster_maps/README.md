# Head-group cluster maps

Prebuilt head-group cluster maps consumed at runtime. Each map assigns every
`(layer, kv_head)` to a `(cluster, column)` so heads with similar retention
budgets — possibly across different layers — share physical KV blocks, removing
the head-group max-pool waste. See
`../../../tangram_impl/tangram-asp/17-head-group-clustering/HANDOFF.md` for the
design.

## How a map is selected (`head_group_cluster_map`)

The `head_group_cluster_map` config (CLI `--head-group-cluster-map`, LLM kwarg
`head_group_cluster_map=...`) takes one of three forms, resolved once at engine
startup (`CacheConfig.resolve_head_group_cluster_map`):

- **`None` (default) — auto-resolve.** When compression is enabled, the engine
  looks up the bundled map matching the running model, `compression_scorer`,
  `page_group_size`, and `compression_level` under this directory and uses it;
  if none matches (or under TP>1, or for a scorer/level that ships no map) it
  falls back to the identity map (adjacent-head grouping) with a warning. The
  lookup follows the naming convention below; the map's `meta` is cross-checked
  against the model (scope, source model, `page_group_size`, `num_kv_heads`)
  before it is accepted, so a slug collision cannot silently load a wrong map.
- **An explicit `.npz` path** — load that map strictly (a missing or mismatched
  file raises). Bypasses auto-resolution.
- **The literal `"identity"`** — force the identity map (adjacent-head grouping
  within a layer, i.e. the original head-group paging); opt out of
  auto-resolution.

Add new maps by dropping a file that follows the convention below — auto-resolve
picks it up with no code change. The resolver derives the directory from the
installed package; set `TANGRAM_CLUSTER_MAPS_DIR` to point at a relocated tree.

## Naming convention

```
<method>/<model-id>/profile.npz                  # retention profile, cross-layer threshold scope (crosslayer_head)
<method>/<model-id>/profile_perlayer.npz         # retention profile, per-layer threshold scope (perlayer_head)
<method>/<model-id>/pg<g>_r<ratio>.npz           # cross-layer cluster map (global scope) <- profile.npz
<method>/<model-id>/pg<g>_r<ratio>_perlayer.npz  # per-layer cluster map (per_layer scope) <- profile_perlayer.npz
```

- `method`: the compression scorer whose retention the clustering is built from —
  `fastkvzip`, `ea` (ExpectedAttention), `keydiff`, or `snapkv`. Different scorers
  keep different heads, so each gets its own map.
- `model-id`: lowercase short model id (e.g. `qwen3-4b-instruct-2507`).
- `page_group_size` (`g`): cluster size, listed first (must match the runtime
  `--page-group-size`).
- `base_ratio`: the compression ratio whose per-head retention ranking the
  clustering was built from. The map encodes a *ranking* and is valid across all
  ratios (the runtime reads only `cluster_of` / `column_of` / `page_group_size`,
  never `base_ratio`); this field only records which ratio's profile produced it.

Which map a level needs is set by the level's threshold SCOPE, not by its
calibration granularity (head vs cluster — both read the same `cluster_of` /
`column_of`):

- The plain variant clusters **across layers** (global scope) and pairs the
  cross-layer-threshold levels: `crosslayer_head` (head-calibrated) and
  `crosslayer_cluster` (cluster-calibrated). Both threshold over all layers at
  once, so cluster ids need not encode a layer.
- The `_perlayer` variant clusters **within each layer** (per-layer scope) and
  pairs the per-layer-threshold levels: `perlayer_head` (head-calibrated) and
  `perlayer_cluster` (cluster-calibrated). Both threshold within each layer,
  which requires `cluster_id // num_groups` to be the physical layer — only the
  within-layer clustering guarantees that.

The two maps come from two different profiles (`profile.npz` vs
`profile_perlayer.npz`) because the measured retention itself depends on the
threshold scope. A per-layer level paired with a cross-layer map buckets
unrelated clusters by `cluster_id // num_groups` (and pools disparate
cross-layer score scales) and is incorrect — keep the scope pairing.

Add new maps by dropping a file that follows this scheme — no code change needed.

## Available maps

Built for four TP=1 models × four scorers × both scopes = 32 maps (plus the two
sibling profiles per model/scorer). Every `<method>/<model-id>/` directory holds
exactly four files: `profile.npz`, `profile_perlayer.npz`, `pg4_r0.3.npz`,
`pg4_r0.3_perlayer.npz`. `page_group_size = 4` matches the serving / RULER-sweep
config for all four models.

For SWA models (gemma-3, gpt-oss) clustering covers the **static (full-attention)
layers only** — the sliding-window layers are not compressed — so the layer count
below is the static-layer count, not the model's total depth.

| model | scorers | static layers × KV heads = members |
|---|---|---|
| `qwen3-4b-instruct-2507` | fastkvzip, ea, keydiff, snapkv | 36 × 8 = 288 |
| `llama-3.1-8b-instruct` | fastkvzip, ea, keydiff, snapkv | 32 × 8 = 256 |
| `gemma-3-12b-it` (SWA) | fastkvzip, ea, keydiff, snapkv | 8 × 8 = 64 |
| `gpt-oss-20b` (SWA) | fastkvzip, ea, keydiff, snapkv | 12 × 8 = 96 |

`qwen3-30b-a3b-instruct-2507` ships only `ea/.../profile.npz`: it serves at TP=2,
so its per-rank head layout needs a TP-aware map that is not yet built.

The sibling `profile.npz` / `profile_perlayer.npz` are the committed per-head
retention profiles (stage-1 output) each map was clustered from; keep them so
maps can be rebuilt or re-clustered at a different `pg` without re-running the
GPU profiling pass.

## Schema (`.npz`, validated by `head_grouped_layout.load_cluster_map`)

| key | shape | meaning |
|---|---|---|
| `cluster_of` | `[num_layers, num_kv_heads]` int | (layer, head) → cluster id |
| `column_of` | `[num_layers, num_kv_heads]` int | column within the cluster (0..g-1) |
| `cluster_members` | `[num_clusters, g, 2]` int | cluster → [(layer, head), ...] (inverse) |
| `rank_score` | `[num_layers, num_kv_heads]` float | retention score used for clustering |
| `page_group_size` | scalar | cluster size `g` |
| `base_ratio` | scalar | ratio whose profile produced the ranking |
| `meta` | json bytes | model, sample config, rank-stability stats |

Invariant: the `(cluster, column)` assignment is a bijection — every
`(layer, head)` occupies exactly one column of one cluster, and every cluster is
full to `page_group_size`.

## Regenerating / adding a model

The whole tree is regenerated by two batch drivers (self-contained — they need
only HuggingFace model ids and, for `fastkvzip`, a gate from the Hub):

```bash
# Stage 1: all profiles (GPU, 8-way). Two threshold scopes per (model, scorer):
#   crosslayer_head -> profile.npz, perlayer_head -> profile_perlayer.npz.
bash tools/head_group_clustering/build_all_profiles.sh

# Stage 2: all cluster maps (CPU, fast). global <- profile.npz,
#   per_layer <- profile_perlayer.npz; page_group_size 4.
bash tools/head_group_clustering/build_all_maps.sh
```

For one model/scorer by hand — note the level/scope pairing:

```bash
# profile at a threshold scope (GPU). --scorer selects the method (fastkvzip
# default / keydiff / snapkv / expected_attention; expected_attention lives under
# ea/). --level perlayer_head would instead write profile_perlayer.npz.
CUDA_VISIBLE_DEVICES=<idle> python -m tools.head_group_clustering.build_profile \
  --model <hf-model-id> --scorer expected_attention --level crosslayer_head \
  --out tools/head_group_clustering/cluster_maps/ea/<model-id>/profile.npz

# cluster it into a map (CPU); --cluster-scope global pairs the crosslayer levels.
python -m tools.head_group_clustering.build_cluster_map \
  --profile tools/head_group_clustering/cluster_maps/ea/<model-id>/profile.npz \
  --base-ratio 0.3 --page-group-size 4 --cluster-scope global \
  --out tools/head_group_clustering/cluster_maps/ea/<model-id>/pg4_r0.3.npz
```

`build_profile.py` measures retention by reading the live engine's own keep
decision, so a fresh profile tracks exactly what the engine evicts at run time
rather than a numpy re-derivation of it. The engine path was cross-checked
against the earlier transformers-replay profiler on qwen3-4b: per-head retention
ranks at Spearman ~0.999 and yields the same cluster map up to a handful of
boundary heads (co-clustering ~99.9% per-layer) — close but not byte-identical,
the expected difference between reading the engine directly and replaying the
model.
