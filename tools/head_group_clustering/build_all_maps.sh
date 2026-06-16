#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Build every head-group cluster map from the retention profiles that
# build_all_profiles.sh produced. CPU-only and fast (no GPU, no engine).
#
# One map per profile, scope paired to the profile's threshold scope:
#   profile.npz          (crosslayer_head) --cluster-scope global    -> pg4_r0.3.npz
#   profile_perlayer.npz (perlayer_head)   --cluster-scope per_layer -> pg4_r0.3_perlayer.npz
# The level/scope pairing is required: a per-layer selection level thresholds
# within each layer, so its map must cluster within-layer (clustering across
# layers would pool disparate cross-layer score scales).
#
# page_group_size = 4 for all four TP=1 models (matches the serving / RULER
# sweep config). Maps are ratio-independent (the runtime reads only
# cluster_of / column_of / page_group_size, never base_ratio), so a single
# r0.3 map serves every compression ratio; the "_r0.3" in the name records
# which retention snapshot it was clustered on, nothing more.
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CMAPS="$ROOT/tools/head_group_clustering/cluster_maps"
PG="${PG:-4}"
RATIO="${RATIO:-0.3}"

MODELS=(qwen3-4b-instruct-2507 llama-3.1-8b-instruct gemma-3-12b-it gpt-oss-20b)
SCORERS=(fastkvzip keydiff snapkv ea)

build_one() {
  local profile="$1" scope="$2" out="$3"
  PYTHONPATH="$ROOT" python -m tools.head_group_clustering.build_cluster_map \
    --profile "$profile" --out "$out" \
    --base-ratio "$RATIO" --page-group-size "$PG" --cluster-scope "$scope"
}

n_ok=0; n_fail=0
for sc in "${SCORERS[@]}"; do
  for mo in "${MODELS[@]}"; do
    dir="$CMAPS/$sc/$mo"
    # cross-layer map <- crosslayer_head profile (global scope)
    if build_one "$dir/profile.npz" global "$dir/pg${PG}_r${RATIO}.npz"; then
      n_ok=$((n_ok + 1)); else n_fail=$((n_fail + 1)); echo "[FAIL] $sc/$mo global"; fi
    # per-layer map <- perlayer_head profile (per_layer scope)
    if build_one "$dir/profile_perlayer.npz" per_layer "$dir/pg${PG}_r${RATIO}_perlayer.npz"; then
      n_ok=$((n_ok + 1)); else n_fail=$((n_fail + 1)); echo "[FAIL] $sc/$mo per_layer"; fi
  done
done
echo "[maps] built $n_ok, failed $n_fail (expected 32)"
