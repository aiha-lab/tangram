#!/usr/bin/env bash
# E2E speedup of Tangram KV-cache compression: r=1.0 (uncompressed) vs r=0.25/0.1,
# wall-clock generation time on an SCBench task. Thin shim over the real harness
# benchmark_performance.sh (single-turn, exact token budget -> apples-to-apples).
# Defaults reproduce snapkv + fastkvzip / perlayer_cluster / scbench_vt / Qwen3-4B
# (~2x). Override any knob via env.
#
# Each configuration runs once per execution mode (GRAPH_MODES):
#   eager — enforce_eager, the historical measurement mode
#   graph — VLLM_COMPILE + piecewise CUDA graphs (TANGRAM_GRAPH=1; head-grouped
#           runs are pinned to PIECEWISE at config time)
# The summary reports, per scorer x mode, the compression speedup vs r=1.0 and,
# per scorer x ratio, the graph-vs-eager gain.
#
# NOTE on the r=1.0 reference: it runs WITH head-grouped paging (PAGE_GROUP_SIZE,
# default 4), so the reported speedup is compression vs uncompressed-Tangram —
# NOT vs vanilla vLLM. Vanilla runs page_group_size=None and is substantially
# faster at long context (measured 3.08x vs pg=4 uncompressed at ~125k ctx);
# use it when comparing against upstream, not this script's reference column.
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
GRAPH_MODES=${GRAPH_MODES:-"eager graph"}   # subset to skip a mode, e.g. "graph"

MODEL_KEY=$(basename "${MODEL}" | tr 'A-Z' 'a-z')

# ---- What this does ------------------------------------------------------
cat <<EOF
==========================================================================
 Tangram speedup quick-reproduce
 Measures e2e generation-time speedup of KV-cache compression:
   speedup = time(r=1.0, uncompressed) / time(r=0.25, compressed)
 and the execution-mode gain: time(eager) / time(graph) per ratio.
 Scorers : ${SCORERS}
 Modes   : ${GRAPH_MODES}
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

    for MODE in ${GRAPH_MODES}; do
        case "${MODE}" in
            eager) TANGRAM_GRAPH_VALUE=0 ;;
            graph) TANGRAM_GRAPH_VALUE=1 ;;
            *) echo "Unknown mode '${MODE}' in GRAPH_MODES (use eager|graph)" >&2
               exit 1 ;;
        esac

        echo ""
        echo ">>> scorer=${SCORER}  level=${LEVEL}  mode=${MODE}"
        TANGRAM_GRAPH="${TANGRAM_GRAPH_VALUE}" \
        GPU_ID="${GPU_ID}" MODEL="${MODEL}" MAX_LEN="${MAX_LEN}" \
        SCORER="${SCORER}" LEVEL="${LEVEL}" DATASET="${DATASET}" RATIOS="${RATIOS}" \
        PAGE_GROUP_SIZE="${PAGE_GROUP_SIZE}" MAX_TOKENS="${MAX_TOKENS}" NUM="${NUM}" \
        HEAD_GROUP_CLUSTER_MAP="${MAP}" \
        OUTPUT_DIR="${OUTPUT_DIR:-${DIR}/performance_results}/${SCORER}_${LEVEL}_${MODE}" \
            bash "${DIR}/benchmark_performance.sh" >/dev/null
        echo "    done."
    done
done

# ---- Consolidated summary ------------------------------------------------
# Two tables: (1) per scorer x mode, per-ratio wall-clock + compression speedup
# vs r=1.0; (2) per scorer x ratio, the graph-vs-eager gain.
BASE="${OUTPUT_DIR:-${DIR}/performance_results}"
SCORERS="${SCORERS}" LEVEL="${LEVEL}" RATIOS="${RATIOS}" DATASET="${DATASET}" \
GRAPH_MODES="${GRAPH_MODES}" PAGE_GROUP_SIZE="${PAGE_GROUP_SIZE}" \
    python3 - "${BASE}" <<'PY'
import json, os, sys

base = sys.argv[1]
scorers = os.environ["SCORERS"].split()
level = os.environ["LEVEL"]
dataset = os.environ["DATASET"]
modes = os.environ["GRAPH_MODES"].split()
pg = os.environ["PAGE_GROUP_SIZE"]
ratios = [float(r) for r in os.environ["RATIOS"].split()]
ref = max(ratios)                                  # r=1.0 reference
comp = [r for r in ratios if r != ref]

def elapsed(scorer, mode, ratio):
    d = os.path.join(base, f"{scorer}_{level}_{mode}", dataset)
    if not os.path.isdir(d):
        return None
    for fn in os.listdir(d):
        if (fn.endswith(f"_r{ratio:g}_pg{pg}.json")
                or fn.endswith(f"_r{ratio}_pg{pg}.json")):
            j = json.load(open(os.path.join(d, fn)))
            b = j.get("benchmark", {})
            return b.get("elapsed_sec", j.get("generation_time_sec"))
    return None

w = max(8, max(len(s) for s in scorers))
cols = [f"r{ref:g} (s)"] + [f"r{r:g} (s)" for r in comp] + [f"{r:g}x" for r in comp]
print()
print("=" * 86)
print(f" SPEEDUP SUMMARY   dataset={dataset}  level={level}")
print(f" speedup = time(r={ref:g}) / time(r=compressed), within each mode")
print("=" * 86)
print(f"{'scorer':<{w}}  {'mode':<6}  " + "  ".join(f"{c:>10}" for c in cols))
for s in scorers:
    for m in modes:
        t_ref = elapsed(s, m, ref)
        cells = [f"{t_ref:10.1f}" if t_ref else f"{'--':>10}"]
        for r in comp:
            t = elapsed(s, m, r)
            cells.append(f"{t:10.1f}" if t else f"{'--':>10}")
        for r in comp:
            t = elapsed(s, m, r)
            sp = (t_ref / t) if (t_ref and t) else None
            cells.append(f"{sp:9.2f}x" if sp else f"{'--':>10}")
        print(f"{s:<{w}}  {m:<6}  " + "  ".join(cells))
print("=" * 86)

if "eager" in modes and "graph" in modes:
    print()
    print("=" * 86)
    print(" GRAPH-MODE GAIN   gain = time(eager) / time(graph), per ratio")
    print("=" * 86)
    gcols = [f"r{r:g}" for r in ratios]
    print(f"{'scorer':<{w}}  " + "  ".join(f"{c:>10}" for c in gcols))
    for s in scorers:
        cells = []
        for r in ratios:
            te, tg = elapsed(s, "eager", r), elapsed(s, "graph", r)
            gain = (te / tg) if (te and tg) else None
            cells.append(f"{gain:9.2f}x" if gain else f"{'--':>10}")
        print(f"{s:<{w}}  " + "  ".join(cells))
    print("=" * 86)
PY
