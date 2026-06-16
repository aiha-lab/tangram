#!/usr/bin/env bash
# RULER accuracy across compression ratios for one method.
#
# Sibling of benchmark_scbench.sh — same method-selection knobs (SCORER /
# LEVEL / RESUME) and engine setup, but drives benchmark_ruler.py over RULER's
# synthetic long-context tasks instead of SCBench. RULER adds a context-LENGTH
# sweep axis (4096 / 8192 / 16384); each (length, ratio) is one model load.
#
# Accuracy protocol (RULER reference):
#   * single-turn          — one context+question prompt per sample
#   * natural EOS          — stop at EOS (no --force-exact-tokens)
#   * per-task length      — output budget from the dataset's max_new_tokens
#   * string-match metric  — recall (retrieval/tracking/extraction) or any-match (QA)
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
#   RESUME  = 1 (skip already-saved (length,task,ratio) cells) | 0 (recompute all)
# Results land in results_ruler/<scorer>_<selection>/ so methods stay separate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Force spawn: vLLM V1's engine core forks by default; CUDA touched in the driver
# makes the fork raise "Cannot re-initialize CUDA".
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# Reduce allocator fragmentation so the transient compression score-buffer spike
# can reuse reserved blocks. PyTorch renamed the env var, so set both names (the
# old PYTORCH_CUDA_ALLOC_CONF is deprecated/ignored on current builds).
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- Model ---------------------------------------------------------------
GPU_ID=${GPU_ID:-0}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
# RULER's longest config here is 16384 tokens; 32768 holds it plus generation
# without over-allocating the KV cache / per-member block table for a 262k
# window (which is unnecessary for RULER and a known OOM source on 4B at 0.9).
MAX_LEN=${MAX_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
PYTHON=${PYTHON:-python3}

# ---- Method --------------------------------------------------------------
SCORER=${SCORER:-snapkv}
# Selection level (axis 1): crosslayer_head | perlayer_head | crosslayer_cluster
# | perlayer_cluster | uniform.
LEVEL=${LEVEL:-crosslayer_head}

# ---- Sweep ---------------------------------------------------------------
LENGTHS=${LENGTHS:-"8192 4096 16384"}   # 8K -> 4K -> 16K completion order
RATIOS=${RATIOS:-"1.0 0.7 0.5 0.3"}
TASKS=${TASKS:-}            # empty = all 13 RULER tasks
NUM=${NUM:-50}             # samples PER TASK (RULER ships 500/task)
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}

# ---- Method-specific args ------------------------------------------------
METHOD_ARGS=(--compression-scorer "${SCORER}" --compression-level "${LEVEL}")
SELECTION="${LEVEL}"

# RESUME=1 skips (length, task, ratio) cells already saved under OUTPUT_DIR, so
# an interrupted sweep continues with the same command (fully-done lengths skip
# the model load too). Default 0 recomputes everything.
if [ "${RESUME:-0}" = "1" ]; then
    METHOD_ARGS+=(--skip-existing)
fi

if [ -n "${TASKS}" ]; then
    METHOD_ARGS+=(--tasks "${TASKS}")
fi

# LOG_STATS=1 turns on vLLM's periodic engine stats logger (prints the
# cumulative "Preemptions: N" line when N>0).
if [ "${LOG_STATS:-0}" = "1" ]; then
    METHOD_ARGS+=(--enable-log-stats)
fi

# Compression keep-geometry overrides (the engine/bench defaults are tuned for
# SCBench's long contexts: window 4096 / floor 512). For RULER's short contexts
# those floors swamp the ratio, so set them small (e.g. WINDOW_SIZE=32 FLOOR_MIN=0)
# to make the compression ratio the binding KV budget. Only applied when set.
if [ -n "${WINDOW_SIZE:-}" ]; then
    METHOD_ARGS+=(--compression-window-size "${WINDOW_SIZE}")
fi
if [ -n "${FLOOR_MIN:-}" ]; then
    METHOD_ARGS+=(--compression-floor-min "${FLOOR_MIN}")
fi

case "${SCORER}" in
    fastkvzip)
        # Gate-based. GATE_PATH overrides the gate checkpoint (absolute path or
        # a Hub-relative name). Unset → the engine auto-resolves "fastkvzip".
        DEFAULT_PG=4
        if [ -n "${GATE_PATH:-}" ]; then
            METHOD_ARGS+=(--compression-gate-path "${GATE_PATH}")
        fi
        ;;
    snapkv)
        DEFAULT_PG=4
        METHOD_ARGS+=(--compression-snap-window "${SNAP_WINDOW:-32}"
                      --compression-snap-kernel "${SNAP_KERNEL:-7}")
        ;;
    keydiff|streamingllm|tova|expected_attention)
        DEFAULT_PG=4
        ;;
    *)
        echo "Unknown SCORER='${SCORER}' (use fastkvzip|snapkv|keydiff|streamingllm|tova|expected_attention)" >&2
        exit 1
        ;;
esac

# Head-group cluster map (applies to ANY scorer). Set HEAD_GROUP_CLUSTER_MAP to
# a map .npz (see tools/head_group_clustering); a missing/unset path falls back
# to identity (adjacent-head) grouping.
if [ -n "${HEAD_GROUP_CLUSTER_MAP:-}" ] && [ -f "${HEAD_GROUP_CLUSTER_MAP}" ]; then
    METHOD_ARGS+=(--head-group-cluster-map "${HEAD_GROUP_CLUSTER_MAP}")
fi
PAGE_GROUP_SIZE=${PAGE_GROUP_SIZE:-${DEFAULT_PG}}
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/results_ruler/${SCORER}_${SELECTION}"}

# ---- Tensor parallel (opt-in) --------------------------------------------
# TP=1 (default) keeps single-GPU behavior. TP>1 runs tensor-parallel across the
# GPUs listed in GPU_ID (a comma list, e.g. "0,1") and disables the custom
# all-reduce (required by the Tangram TP path).
TP=${TP:-1}
TP_ARGS=()
if [ "${TP}" -gt 1 ]; then
    TP_ARGS=(--tensor-parallel-size "${TP}" --disable-custom-all-reduce)
fi

# ---- Run -----------------------------------------------------------------
# Outer loop over context lengths so each length's full ratio sweep completes
# before the next (the all-task average is valid after every length).
for LENGTH in ${LENGTHS}; do
    for RATIO in ${RATIOS}; do
        echo "===== ${SCORER} ${SELECTION}  length=${LENGTH}  ratio=${RATIO}  tp=${TP} ====="
        CUDA_VISIBLE_DEVICES="${GPU_ID}" "$PYTHON" "${SCRIPT_DIR}/benchmark_ruler.py" \
            -l "${LENGTH}" \
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
done

# ---- Accuracy summary ----------------------------------------------------
# Compact per-(length, task, ratio) avg_score table read back from saved JSON.
echo ""
echo "===== accuracy summary: ${SCORER} ${SELECTION} ====="
"$PYTHON" - "${OUTPUT_DIR}" <<'PY'
import json, os, sys
root = sys.argv[1]
rows = {}      # (length, task) -> {ratio: score}
ratios = set()
for dp, _, files in os.walk(root):
    for fn in files:
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(dp, fn)) as f:
            d = json.load(f)
        length, task, r = d.get("length"), d.get("task"), d.get("ratio")
        if length is None or task is None or r is None:
            continue
        rows.setdefault((str(length), task), {})[r] = d.get("avg_score")
        ratios.add(r)
if not rows:
    print("(no results found under", root, ")")
    sys.exit(0)
ratios = sorted(ratios, reverse=True)
keyw = max(len(f"{ln}/{tk}") for ln, tk in rows)
hdr = "  ".join(f"r{r:<6}" for r in ratios)
print(f"{'length/task':<{keyw}}  {hdr}")
for ln, tk in sorted(rows):
    cells = "  ".join(
        (f"{rows[(ln, tk)][r]*100:5.1f}%" if rows[(ln, tk)].get(r) is not None
         else "   -- ")
        for r in ratios
    )
    print(f"{ln + '/' + tk:<{keyw}}  {cells}")
PY
echo "ALL_DONE"
