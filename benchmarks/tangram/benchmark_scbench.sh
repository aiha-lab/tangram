#!/usr/bin/env bash
# SCBench accuracy for one compression method. Sibling of benchmark_ruler.sh.
#
# Native SCBench protocol: multi-turn (every question per context), natural EOS,
# per-dataset output length. ratio=0 is the uncompressed reference.
#
#   SCORER  = fastkvzip | snapkv | keydiff | streamingllm | tova | expected_attention
#   SCOPE   = layer (default) | global | uniform      # axis 1, see budget_scope.py
#   RESUME  = 1 skips (dataset, setting) cells already saved
#   RATIOS  = evicted fractions (ratio regime) | BUDGETS = fixed KV tokens per
#             (layer, head group) (budget regime). Either, or both.
#
# Results land in results_accuracy/<scorer>_<selection>/ so methods stay separate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# The driver touches CUDA, and a forked engine core cannot re-initialize it.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# Let the transient compression-gather spike reuse reserved blocks.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- Model ---------------------------------------------------------------
GPU_ID=${GPU_ID:-0}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
MAX_LEN=${MAX_LEN:-262144}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
PYTHON=${PYTHON:-python3}

# ---- Method --------------------------------------------------------------
SCORER=${SCORER:-snapkv}
SCOPE=${SCOPE:-layer}

# ---- Sweep ---------------------------------------------------------------
DATASET=${DATASET:-mid}
RATIOS=${RATIOS-"0.0 0.3 0.5 0.7"}   # ${VAR-...}: RATIOS="" means "no ratio sweep"
BUDGETS=${BUDGETS:-}
NUM=${NUM:-100}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}

# ---- Budget-regime knobs (ignored when BUDGETS is empty) -----------------
# 1 = the fresh chunk competes for eviction too; 0 measured better.
EVICT_CURRENT_CHUNK=${EVICT_CURRENT_CHUNK:-0}
# persist holds the score a ratio run would rank — an ablation, not a knob.
SLOT_SCORE_SOURCE=${SLOT_SCORE_SOURCE:-auto}
# A budget must exceed sink + the protected tail, and the tail is the whole
# fresh chunk, so a small budget needs both of these small (e.g. 512 / 32).
# The bench defaults (8192 / 4096) are tuned for SCBench's long contexts.
CHUNK_SIZE=${CHUNK_SIZE:-}
WINDOW_SIZE=${WINDOW_SIZE:-}

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
# fully-done cell skips the model load too.
flag --skip-existing "${RESUME:-0}"
# fastkvzip only; unset lets the engine auto-resolve the gate.
opt --compression-gate-path "${GATE_PATH:-}"
opt --compression-scorer-options "${SCORER_OPTIONS:-}"
opt --compression-chunk-size "${CHUNK_SIZE}"
opt --compression-window-size "${WINDOW_SIZE}"
# A .npz from tools/head_group_clustering; unset or missing falls back to
# identity (adjacent-head) grouping.
if [ -n "${HEAD_GROUP_CLUSTER_MAP:-}" ] && [ -f "${HEAD_GROUP_CLUSTER_MAP}" ]; then
    METHOD_ARGS+=(--head-group-cluster-map "${HEAD_GROUP_CLUSTER_MAP}")
fi

# Budget-regime policy, plus the tag that keeps result files apart for runs
# differing only in it (or in a scorer setting, a different algorithm).
BUDGET_ARGS=()
TAG_PARTS=()
if [ -n "${SCORER_OPTIONS:-}" ]; then
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
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/results_accuracy/${SCORER}_${SELECTION}"}

# ---- Run -----------------------------------------------------------------
run_one() {
    # $@ = the setting-specific args (a ratio, or a budget plus its policy).
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "$PYTHON" "${SCRIPT_DIR}/benchmark_scbench.py" \
        -d "${DATASET}" \
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

for RATIO in ${RATIOS}; do
    echo "===== ${SCORER} ${SELECTION}  dataset=${DATASET}  ratio=${RATIO}" \
         "options=${SCORER_OPTIONS:-<defaults>}  tp=${TP} ====="
    run_one --compression-ratio "${RATIO}" \
            ${RATIO_TAG:+--tag "${RATIO_TAG}"}
done

for BUDGET in ${BUDGETS}; do
    echo "===== ${SCORER} ${SELECTION}  dataset=${DATASET}  budget=${BUDGET}" \
         "evict_current_chunk=${EVICT_CURRENT_CHUNK}" \
         "slot_score_source=${SLOT_SCORE_SOURCE}" \
         "options=${SCORER_OPTIONS:-<defaults>}  tp=${TP} ====="
    run_one --compression-budget-tokens "${BUDGET}" \
            ${BUDGET_TAG:+--tag "${BUDGET_TAG}"} "${BUDGET_ARGS[@]}"
done

# ---- Accuracy summary ----------------------------------------------------
# Compact per-(dataset, setting) avg_score table read back from the JSON.
echo ""
echo "===== accuracy summary: ${SCORER} ${SELECTION} ====="
"$PYTHON" - "${OUTPUT_DIR}" <<'PY'
import json, os, sys
root = sys.argv[1]
rows = {}      # dataset -> {setting: score}
settings = set()
for dp, _, files in os.walk(root):
    for fn in files:
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(dp, fn)) as f:
            d = json.load(f)
        ds, r = d.get("dataset"), d.get("ratio")
        if ds is None or r is None:
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
        rows.setdefault(ds, {})[setting] = d.get("avg_score")
        settings.add(setting)
if not rows:
    print("(no results found under", root, ")")
    sys.exit(0)

def _setting_key(label):
    """Ratio settings first (ascending evicted fraction, baseline 0 leading),
    then budgets (descending). The two are different retention targets and are
    not comparable by label alone."""
    head = label.split("+")[0]
    if head.startswith("ratio"):
        return (0, float(head[len("ratio"):]))
    return (1, -float(head[len("budget"):]))

settings = sorted(settings, key=_setting_key)
w = max(len(d) for d in rows)
colw = max(7, max(len(x) for x in settings))
hdr = "  ".join(f"{x:<{colw}}" for x in settings)
print(f"{'dataset':<{w}}  {hdr}")
for ds in sorted(rows):
    cells = "  ".join(
        (f"{rows[ds][x]*100:{colw}.1f}" if rows[ds].get(x) is not None
         else f"{'--':>{colw}}")
        for x in settings
    )
    print(f"{ds:<{w}}  {cells}")
PY
echo "ALL_DONE"
