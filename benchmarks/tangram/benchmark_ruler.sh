#!/usr/bin/env bash
# RULER accuracy for one compression method. Sibling of benchmark_scbench.sh;
# adds a context-LENGTH sweep. Each (length, setting) is one model load.
#
# RULER reference protocol: single-turn, natural EOS, per-task output length,
# string-match metric. ratio=0 is the uncompressed reference.
#
#   SCORER  = fastkvzip | snapkv | keydiff | streamingllm | tova | expected_attention
#   SCOPE   = layer (default) | global | uniform      # axis 1, see budget_scope.py
#   RESUME  = 1 skips (length,task,setting) cells already saved
#
# Results land in results_ruler/<scorer>_<selection>/ so methods stay separate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# The driver touches CUDA, and a forked engine core cannot re-initialize it.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# Let the transient compression spike reuse reserved blocks. Both names: the
# CUDA-prefixed one is deprecated but still what older builds read.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- Model ---------------------------------------------------------------
GPU_ID=${GPU_ID:-0}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
# Fits RULER's longest (16384) plus generation. A 262k window over-allocates
# the block table and OOMs a 4B at 0.85.
MAX_LEN=${MAX_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
PYTHON=${PYTHON:-python3}

# ---- Method --------------------------------------------------------------
SCORER=${SCORER:-snapkv}
SCOPE=${SCOPE:-layer}

# ---- Sweep ---------------------------------------------------------------
LENGTHS=${LENGTHS:-"8192 4096 16384"}   # 8K -> 4K -> 16K completion order
# ${VAR-...}, not :-, so RATIOS="" means "no ratio sweep".
RATIOS=${RATIOS-"0.0 0.3 0.5 0.7"}
# Fixed KV tokens per (layer, head group) — a different retention target, not a
# ratio. Comparable to the ratio keeping the same amount: b ~= (1-r) * length.
BUDGETS=${BUDGETS:-}
# Budget runs only. 1 = the fresh chunk competes too; 0 measured better.
EVICT_CURRENT_CHUNK=${EVICT_CURRENT_CHUNK:-0}
# Budget runs only. persist holds the score a ratio run would rank, which is
# what separates the retention target from the score in a ratio->budget
# comparison. An ablation, not a serving setting.
SLOT_SCORE_SOURCE=${SLOT_SCORE_SOURCE:-auto}
# Settings the SCORER declares, key=value,key=value. Both regimes.
#   SCORER_OPTIONS=anchor=normalized  -> KeyDiff Eq.(8) as written, mu(K-hat)
#   unset                             -> mu(K), the paper's experiments
SCORER_OPTIONS=${SCORER_OPTIONS:-}
TASKS=${TASKS:-}            # empty = all 13 RULER tasks
NUM=${NUM:-50}             # samples PER TASK (RULER ships 500/task)
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}

# ---- Args ----------------------------------------------------------------
SELECTION="${SCOPE}"
PAGE_GROUP_SIZE=${PAGE_GROUP_SIZE:-4}
METHOD_ARGS=(--compression-scorer "${SCORER}"
             --compression-budget-scope "${SCOPE}")

# Append "<flag> <value>" only when the value is set; append a bare flag when
# its variable is 1. An unknown SCORER needs no check here — argparse rejects it
# against the same list, and CacheConfig against the scorer registry.
opt()  { if [ -n "${2:-}" ]; then METHOD_ARGS+=("$1" "$2"); fi; }
flag() { if [ "${2:-0}" = "1" ]; then METHOD_ARGS+=("$1"); fi; }

# RESUME: an interrupted sweep continues with the same command, and a
# fully-done length skips the model load too.
flag --skip-existing "${RESUME:-0}"
# vLLM's periodic stats logger — the "Preemptions: N" line.
flag --enable-log-stats "${LOG_STATS:-0}"
opt --tasks "${TASKS:-}"
# A KV pool too small to hold every admitted request forces preemption.
opt --num-gpu-blocks-override "${NUM_GPU_BLOCKS:-}"
# fastkvzip only; unset lets the engine auto-resolve the gate.
opt --compression-gate-path "${GATE_PATH:-}"
# The bench defaults (chunk 8192 / window 4096 / floor 512 / sink 32) are tuned
# for SCBench's long contexts and swamp the target at RULER's lengths. A budget
# below the protected tail is unreachable, so a budget sweep sets a small chunk
# and window. N_SINK=0 matches a reference that protects no prefix (KeyDiff's).
opt --compression-chunk-size "${CHUNK_SIZE:-}"
opt --compression-window-size "${WINDOW_SIZE:-}"
opt --compression-floor-min "${FLOOR_MIN:-}"
opt --compression-n-sink-tokens "${N_SINK:-}"
opt --compression-scorer-options "${SCORER_OPTIONS}"
# A .npz from tools/head_group_clustering; unset or missing falls back to
# identity (adjacent-head) grouping.
if [ -n "${HEAD_GROUP_CLUSTER_MAP:-}" ] && [ -f "${HEAD_GROUP_CLUSTER_MAP}" ]; then
    METHOD_ARGS+=(--head-group-cluster-map "${HEAD_GROUP_CLUSTER_MAP}")
fi

# Budget-regime policy, plus the tag that keeps result files apart for runs
# differing only in it (or in a scorer setting, a different algorithm).
BUDGET_ARGS=()
TAG_PARTS=()
if [ -n "${SCORER_OPTIONS}" ]; then
    TAG_PARTS+=("$(echo "${SCORER_OPTIONS}" | tr '=,' '-_')")
fi
RATIO_TAG=$(IFS=- ; echo "${TAG_PARTS[*]:-}")
if [ "${EVICT_CURRENT_CHUNK}" = "1" ]; then
    BUDGET_ARGS+=(--compression-evict-current-chunk)
    TAG_PARTS+=("evict-current-chunk")
fi
if [ "${SLOT_SCORE_SOURCE}" != "auto" ]; then
    BUDGET_ARGS+=(--compression-slot-score-source "${SLOT_SCORE_SOURCE}")
    TAG_PARTS+=("${SLOT_SCORE_SOURCE}-scores")
fi
BUDGET_TAG=$(IFS=- ; echo "${TAG_PARTS[*]:-}")

# TP>1 spans the GPUs in GPU_ID ("0,1") and must disable the custom all-reduce.
TP=${TP:-1}
TP_ARGS=()
if [ "${TP}" -gt 1 ]; then
    TP_ARGS=(--tensor-parallel-size "${TP}" --disable-custom-all-reduce)
fi
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/results_ruler/${SCORER}_${SELECTION}"}

# ---- Run -----------------------------------------------------------------
# Length outermost, so the all-task average is valid after every length.
run_one() {
    # $@ = the setting-specific args (a ratio, or a budget).
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "$PYTHON" "${SCRIPT_DIR}/benchmark_ruler.py" \
        -l "${LENGTH}" \
        --num "${NUM}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --page-group-size "${PAGE_GROUP_SIZE}" \
        "${TP_ARGS[@]}" \
        "${METHOD_ARGS[@]}" \
        "$@" \
        -m "${MODEL}" \
        --max-model-len "${MAX_LEN}" \
        --output-dir "${OUTPUT_DIR}"
}

for LENGTH in ${LENGTHS}; do
    for RATIO in ${RATIOS}; do
        echo "===== ${SCORER} ${SELECTION}  length=${LENGTH}  ratio=${RATIO}" \
             "options=${SCORER_OPTIONS:-<defaults>}  tp=${TP} ====="
        run_one --compression-ratio "${RATIO}" \
                ${RATIO_TAG:+--tag "${RATIO_TAG}"}
    done
    for BUDGET in ${BUDGETS}; do
        echo "===== ${SCORER} ${SELECTION}  length=${LENGTH}  budget=${BUDGET}" \
             "evict_current_chunk=${EVICT_CURRENT_CHUNK}" \
             "slot_score_source=${SLOT_SCORE_SOURCE}" \
             "options=${SCORER_OPTIONS:-<defaults>}  tp=${TP} ====="
        run_one --compression-budget-tokens "${BUDGET}" \
                ${BUDGET_TAG:+--tag "${BUDGET_TAG}"} "${BUDGET_ARGS[@]}"
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
        # Every knob that makes a run a different experiment belongs in the
        # label; two experiments sharing a column silently overwrite each other.
        budget = d.get("budget_tokens")
        setting = f"ratio{r}" if budget is None else f"budget{budget}"
        if d.get("evict_current_chunk"):
            setting += "+evict-current-chunk"
        source = d.get("slot_score_source")
        if source and source != "auto":
            setting += f"+{source}"
        if d.get("scorer_options"):
            setting += f"+{d['scorer_options']}"
        rows.setdefault((str(length), task), {})[setting] = d.get("avg_score")
        ratios.add(setting)
if not rows:
    print("(no results found under", root, ")")
    sys.exit(0)
def _setting_key(label: str) -> tuple[int, float]:
    """Sort ratio settings first (ascending evicted fraction, baseline 0
    leading), then budgets (descending). The two are different retention
    targets and are not comparable by label alone."""
    head = label.split("+")[0]
    if head.startswith("ratio"):
        return (0, float(head[len("ratio"):]))
    return (1, -float(head[len("budget"):]))

ratios = sorted(ratios, key=_setting_key)
keyw = max(len(f"{ln}/{tk}") for ln, tk in rows)
width = max(len(r) for r in ratios) + 1
hdr = "  ".join(f"{r:<{width}}" for r in ratios)
print(f"{'length/task':<{keyw}}  {hdr}")
for ln, tk in sorted(rows):
    cells = "  ".join(
        (f"{rows[(ln, tk)][r]*100:>{width}.1f}"
         if rows[(ln, tk)].get(r) is not None else f"{'--':>{width}}")
        for r in ratios
    )
    print(f"{ln + '/' + tk:<{keyw}}  {cells}")
PY
echo "ALL_DONE"
