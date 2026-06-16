#!/usr/bin/env bash
# SCBench accuracy across compression ratios for one method.
#
# Accuracy protocol (native SCBench):
#   * multi-turn          — every question per context (no --single-turn)
#   * natural EOS          — stop at EOS (no --force-exact-tokens)
#   * per-dataset length   — output length from scbench_local.set_gen_length
#
# ratio=1.0 is the uncompressed reference; uniform vs non-uniform differ only
# at ratio<1.0.
#
# Select the method with two knobs:
#   SCORER  = fastkvzip | snapkv | keydiff | streamingllm | tova | expected_attention
#   LEVEL   = crosslayer_head (cross-layer global threshold, head-calibrated; default)
#           | perlayer_head (per-layer threshold, AdaKV-style, head-calibrated;
#             the validated pairing for SCORER=expected_attention)
#           | crosslayer_cluster (cross-layer threshold, cluster-calibrated;
#             exact budget, needs a global cluster map)
#           | perlayer_cluster (per-layer threshold, cluster-calibrated;
#             exact per-layer budget, needs a per-layer cluster map)
#           | uniform (same kept count per (layer, group))
#   RESUME  = 1 (skip already-saved (dataset,ratio) cells) | 0 (recompute all)
# Results land in results_accuracy/<scorer>_<selection>/ so methods stay separate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Force spawn: vLLM V1's engine core forks by default; CUDA touched in the driver
# makes the fork raise "Cannot re-initialize CUDA".
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# Reduce allocator fragmentation so the transient compression-gather spike on
# long-context datasets (e.g. repoqa) can reuse reserved-but-unallocated blocks.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- Model ---------------------------------------------------------------
GPU_ID=${GPU_ID:-0}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
MAX_LEN=${MAX_LEN:-262144}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
PYTHON=${PYTHON:-python3}

# ---- Method --------------------------------------------------------------
SCORER=${SCORER:-snapkv}
# Selection level (axis 1): crosslayer_head | perlayer_head | crosslayer_cluster
# | perlayer_cluster | uniform.
LEVEL=${LEVEL:-crosslayer_head}

# ---- Sweep ---------------------------------------------------------------
DATASET=${DATASET:-mid}
RATIOS=${RATIOS:-"1.0 0.7 0.5 0.3"}
NUM=${NUM:-100}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}

# ---- Method-specific args ------------------------------------------------
METHOD_ARGS=(--compression-scorer "${SCORER}" --compression-level "${LEVEL}")
SELECTION="${LEVEL}"

# RESUME=1 skips (dataset, ratio) cells already saved under OUTPUT_DIR, so an
# interrupted sweep continues with the same command (fully-done ratios skip the
# model load too). Default 0 recomputes everything.
if [ "${RESUME:-0}" = "1" ]; then
    METHOD_ARGS+=(--skip-existing)
fi

case "${SCORER}" in
    fastkvzip)
        # Gate-based. GATE_PATH overrides the gate checkpoint; unset → the engine
        # auto-resolves "fastkvzip". Cluster map applied by the common block below.
        DEFAULT_PG=4
        if [ -n "${GATE_PATH:-}" ]; then
            METHOD_ARGS+=(--compression-gate-path "${GATE_PATH}")
        fi
        ;;
    snapkv)
        # Gate-free; identity adjacency (no cluster map).
        DEFAULT_PG=4
        METHOD_ARGS+=(--compression-snap-window "${SNAP_WINDOW:-32}"
                      --compression-snap-kernel "${SNAP_KERNEL:-7}")
        ;;
    keydiff)
        # Gate-free; identity adjacency (no cluster map).
        DEFAULT_PG=4
        ;;
    streamingllm)
        # Gate-free recency baseline; identity adjacency (no cluster map).
        DEFAULT_PG=4
        ;;
    tova)
        # Gate-free last-query attention (head-uniform); identity adjacency.
        DEFAULT_PG=4
        ;;
    expected_attention)
        # Gate-free analytic expected attention; identity adjacency.
        DEFAULT_PG=4
        ;;
    *)
        echo "Unknown SCORER='${SCORER}' (use fastkvzip|snapkv|keydiff|streamingllm|tova|expected_attention)" >&2
        exit 1
        ;;
esac

# Head-group cluster map (applies to ANY scorer). The runner resolves a
# per-scorer map and exports HEAD_GROUP_CLUSTER_MAP; a missing/sentinel path
# (file does not exist) falls back to identity adjacency.
if [ -n "${HEAD_GROUP_CLUSTER_MAP:-}" ] && [ -f "${HEAD_GROUP_CLUSTER_MAP}" ]; then
    METHOD_ARGS+=(--head-group-cluster-map "${HEAD_GROUP_CLUSTER_MAP}")
fi
PAGE_GROUP_SIZE=${PAGE_GROUP_SIZE:-${DEFAULT_PG}}
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/results_accuracy/${SCORER}_${SELECTION}"}

# ---- Tensor parallel (opt-in) --------------------------------------------
# TP=1 (default) keeps the original single-GPU behavior unchanged. TP>1 runs
# tensor-parallel across the GPUs listed in GPU_ID (a comma list, e.g. "0,1")
# and disables the custom all-reduce (required by the Tangram TP path).
TP=${TP:-1}
TP_ARGS=()
if [ "${TP}" -gt 1 ]; then
    TP_ARGS=(--tensor-parallel-size "${TP}" --disable-custom-all-reduce)
fi

# ---- Run -----------------------------------------------------------------
for RATIO in ${RATIOS}; do
    echo "===== ${SCORER} ${SELECTION}  dataset=${DATASET}  ratio=${RATIO}  tp=${TP} ====="
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "$PYTHON" "${SCRIPT_DIR}/benchmark_scbench.py" \
        -d "${DATASET}" \
        --num "${NUM}" \
        --ratio "${RATIO}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --page-group-size "${PAGE_GROUP_SIZE}" \
        "${TP_ARGS[@]}" \
        "${METHOD_ARGS[@]}" \
        -m "${MODEL}" \
        --max-model-len "${MAX_LEN}" \
        --output-dir "${OUTPUT_DIR}"
done

# ---- Accuracy summary ----------------------------------------------------
# Compact per-(dataset, ratio) avg_score table read back from the saved JSON.
echo ""
echo "===== accuracy summary: ${SCORER} ${SELECTION} ====="
"$PYTHON" - "${OUTPUT_DIR}" <<'PY'
import json, os, sys
root = sys.argv[1]
rows = {}      # dataset -> {ratio: score}
ratios = set()
for dp, _, files in os.walk(root):
    for fn in files:
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(dp, fn)) as f:
            d = json.load(f)
        ds, r = d.get("dataset"), d.get("ratio")
        if ds is None or r is None:
            continue
        rows.setdefault(ds, {})[r] = d.get("avg_score")
        ratios.add(r)
if not rows:
    print("(no results found under", root, ")")
    sys.exit(0)
ratios = sorted(ratios, reverse=True)
w = max(len(d) for d in rows)
hdr = "  ".join(f"r{r:<6}" for r in ratios)
print(f"{'dataset':<{w}}  {hdr}")
for ds in sorted(rows):
    cells = "  ".join(
        (f"{rows[ds][r]*100:5.1f}%" if rows[ds].get(r) is not None else "   -- ")
        for r in ratios
    )
    print(f"{ds:<{w}}  {cells}")
PY
echo "ALL_DONE"
