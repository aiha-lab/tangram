#!/usr/bin/env bash
# E2E speedup of Tangram KV-cache compression: r=1.0 (uncompressed) vs r=0.25/0.1,
# wall-clock generation time on an SCBench task. Thin shim over the real harness
# benchmark_performance.sh (single-turn, exact token budget -> apples-to-apples).
# Defaults reproduce snapkv + fastkvzip / perlayer_cluster / scbench_vt / Qwen3-4B
# (~2x). Override any knob via env.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # benchmarks/tangram
REPO_ROOT="$(cd "${DIR}/../.." && pwd)"

GPU_ID=${GPU_ID:-0}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
SCORERS=${SCORERS:-"snapkv fastkvzip"}   # one run per scorer
LEVEL=${LEVEL:-perlayer_cluster}
DATASET=${DATASET:-scbench_vt}
RATIOS=${RATIOS:-"1.0 0.25 0.1"}
PAGE_GROUP_SIZE=${PAGE_GROUP_SIZE:-4}
MAX_TOKENS=${MAX_TOKENS:-96}
MAX_LEN=${MAX_LEN:-200000}
NUM=${NUM:-10}                # samples (requests); larger -> fuller decode batch

MODEL_KEY=$(basename "${MODEL}" | tr 'A-Z' 'a-z')

# ---- What this does ------------------------------------------------------
cat <<EOF
==========================================================================
 Tangram speedup quick-reproduce
 Measures e2e generation-time speedup of KV-cache compression:
   speedup = time(r=1.0, uncompressed) / time(r=0.25, compressed)
 Scorers : ${SCORERS}
 Level   : ${LEVEL}   Dataset: ${DATASET}   Model: ${MODEL_KEY}   GPU: ${GPU_ID}
==========================================================================
EOF

# Cluster levels need a head-group map; resolve it from the in-repo collection,
# keyed by scorer + model basename so it follows the model (override to relocate).
# perlayer_cluster needs the per-layer map variant (exact per-layer budget); every
# other level uses the cross-layer map.
MAP_SUFFIX=""; [ "${LEVEL}" = "perlayer_cluster" ] && MAP_SUFFIX="_perlayer"

for SCORER in ${SCORERS}; do
    MAP="${REPO_ROOT}/tools/head_group_clustering/cluster_maps/${SCORER}/${MODEL_KEY}/pg${PAGE_GROUP_SIZE}_r0.3${MAP_SUFFIX}.npz"
    [ -f "${MAP}" ] || { echo "Cluster map not found: ${MAP}" >&2; exit 1; }

    echo ""
    echo ">>> scorer=${SCORER}  level=${LEVEL}"
    GPU_ID="${GPU_ID}" MODEL="${MODEL}" MAX_LEN="${MAX_LEN}" \
    SCORER="${SCORER}" LEVEL="${LEVEL}" DATASET="${DATASET}" RATIOS="${RATIOS}" \
    PAGE_GROUP_SIZE="${PAGE_GROUP_SIZE}" MAX_TOKENS="${MAX_TOKENS}" NUM="${NUM}" \
    HEAD_GROUP_CLUSTER_MAP="${MAP}" \
    OUTPUT_DIR="${OUTPUT_DIR:-${DIR}/performance_results}/${SCORER}_${LEVEL}" \
        bash "${DIR}/benchmark_performance.sh" >/dev/null
    echo "    done."
done

# ---- Consolidated summary ------------------------------------------------
# One table across all scorers: per-ratio wall-clock + the speedup vs r=1.0.
BASE="${OUTPUT_DIR:-${DIR}/performance_results}"
SCORERS="${SCORERS}" LEVEL="${LEVEL}" RATIOS="${RATIOS}" DATASET="${DATASET}" \
    python3 - "${BASE}" <<'PY'
import json, os, sys

base = sys.argv[1]
scorers = os.environ["SCORERS"].split()
level = os.environ["LEVEL"]
dataset = os.environ["DATASET"]
ratios = [float(r) for r in os.environ["RATIOS"].split()]
ref = max(ratios)                                  # r=1.0 reference
comp = [r for r in ratios if r != ref]

def elapsed(scorer, ratio):
    d = os.path.join(base, f"{scorer}_{level}", dataset)
    if not os.path.isdir(d):
        return None
    for fn in os.listdir(d):
        if fn.endswith(f"_r{ratio:g}_pg4.json") or fn.endswith(f"_r{ratio}_pg4.json"):
            j = json.load(open(os.path.join(d, fn)))
            b = j.get("benchmark", {})
            return b.get("elapsed_sec", j.get("generation_time_sec"))
    return None

w = max(8, max(len(s) for s in scorers))
cols = [f"r{ref:g} (s)"] + [f"r{r:g} (s)" for r in comp] + [f"{r:g}x" for r in comp]
print()
print("=" * 74)
print(f" SPEEDUP SUMMARY   dataset={dataset}  level={level}")
print(f" speedup = time(r={ref:g}) / time(r=compressed)")
print("=" * 74)
print(f"{'scorer':<{w}}  " + "  ".join(f"{c:>10}" for c in cols))
for s in scorers:
    t_ref = elapsed(s, ref)
    cells = [f"{t_ref:10.1f}" if t_ref else f"{'--':>10}"]
    for r in comp:
        t = elapsed(s, r)
        cells.append(f"{t:10.1f}" if t else f"{'--':>10}")
    for r in comp:
        t = elapsed(s, r)
        sp = (t_ref / t) if (t_ref and t) else None
        cells.append(f"{sp:9.2f}x" if sp else f"{'--':>10}")
    print(f"{s:<{w}}  " + "  ".join(cells))
print("=" * 74)
PY
