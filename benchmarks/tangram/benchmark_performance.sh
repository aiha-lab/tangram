#!/usr/bin/env bash
# SCBench performance (throughput / latency) across compression ratios for one
# method. Sibling of benchmark_scbench.sh — same method-selection knobs, but
# the performance protocol instead of the accuracy protocol.
#
# Performance protocol (apples-to-apples throughput):
#   * single-turn          — context + first question only (--single-turn)
#   * exact token budget   — every request emits MAX_TOKENS (--force-exact-tokens)
#   * fixed --max-tokens   — so decode work is identical across ratios/methods
#
# This isolates the engine cost (prefill + decode + compression overhead) from
# answer-length variance. ratio=0 (evict nothing) is the uncompressed
# reference; uniform vs non-uniform differ only at ratio>0.
#
# Select the method with two knobs:
#   SCORER  = fastkvzip | snapkv | keydiff | streamingllm | tova | expected_attention
#   SCOPE   = layer (per-layer budget, pooled across its head groups; default,
#             needs a per-layer cluster map)
#           | global (one budget pooled across all layers and head groups;
#             needs a global cluster map)
#           | uniform (same kept count per (layer, group))
# Results land in performance_results/<scorer>_<selection>/ so methods stay
# separate, mirroring results_accuracy/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Force spawn: vLLM V1's engine core forks by default; CUDA touched in the driver
# makes the fork raise "Cannot re-initialize CUDA".
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
# Prepend the repo root so this checkout shadows any pip-installed vLLM.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---- Model ---------------------------------------------------------------
GPU_ID=${GPU_ID:-0}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct-1M}
MAX_LEN=${MAX_LEN:-200000}
PYTHON=${PYTHON:-python3}

# ---- Method --------------------------------------------------------------
SCORER=${SCORER:-snapkv}
# Budget scope (axis 1): uniform | layer | global.
SCOPE=${SCOPE:-layer}

# ---- Sweep ---------------------------------------------------------------
DATASET=${DATASET:-scbench_repoqa}
RATIOS=${RATIOS:-"0.0 0.7"}
NUM=${NUM:-10}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_TOKENS=${MAX_TOKENS:-512}
# Fraction of GPU memory the engine may claim; the leftover after weights is the
# KV pool. Raise it when a long-context model cannot fit one full-length request
# (e.g. gemma-3-12b at 124k needs ~46 GiB of KV, just over what 0.90 leaves).
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}

# Page-group size is 4 for every method; PAGE_GROUP_SIZE overrides it (the
# fastkvzip cluster map below must then match the chosen page group).
PAGE_GROUP_SIZE=${PAGE_GROUP_SIZE:-4}

# ---- Method-specific args ------------------------------------------------
METHOD_ARGS=(--compression-scorer "${SCORER}"
             --compression-budget-scope "${SCOPE}")
SELECTION="${SCOPE}"

case "${SCORER}" in
    fastkvzip)
        # Gate-based; the gate auto-resolves from the model. The head-group
        # cluster map (if provided) is applied by the common block below.
        :
        ;;
    snapkv)
        # Gate-free observation-window attention. SnapKV knobs travel through
        # the generic channel: SCORER_OPTIONS="window=32,kernel=7".
        ;;
    keydiff|streamingllm|tova|expected_attention)
        # Gate-free, identity adjacency, no extra arguments (the scorer reads
        # its hyperparameters from the benchmark defaults).
        ;;
    *)
        echo "Unknown SCORER='${SCORER}' (use fastkvzip|snapkv|keydiff|streamingllm|tova|expected_attention)" >&2
        exit 1
        ;;
esac

# Scorer-declared settings as key=value,key=value (see the scorer's OPTIONS).
if [ -n "${SCORER_OPTIONS:-}" ]; then
    METHOD_ARGS+=(--compression-scorer-options "${SCORER_OPTIONS}")
fi

# Head-group cluster map (applies to ANY scorer). The runner resolves a
# per-scorer map and exports HEAD_GROUP_CLUSTER_MAP; a missing/sentinel path
# (file does not exist) falls back to identity adjacency.
if [ -n "${HEAD_GROUP_CLUSTER_MAP:-}" ] && [ -f "${HEAD_GROUP_CLUSTER_MAP}" ]; then
    METHOD_ARGS+=(--head-group-cluster-map "${HEAD_GROUP_CLUSTER_MAP}")
fi
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/performance_results/${SCORER}_${SELECTION}"}

# ---- Run -----------------------------------------------------------------
for RATIO in ${RATIOS}; do
    echo "===== ${SCORER} ${SELECTION}  dataset=${DATASET}  ratio=${RATIO} ====="
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "$PYTHON" "${SCRIPT_DIR}/benchmark_scbench.py" \
        -d "${DATASET}" \
        --num "${NUM}" \
        --compression-ratio "${RATIO}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --page-group-size "${PAGE_GROUP_SIZE}" \
        --max-tokens "${MAX_TOKENS}" \
        --single-turn \
        --force-exact-tokens \
        "${METHOD_ARGS[@]}" \
        -m "${MODEL}" \
        --max-model-len "${MAX_LEN}" \
        --output-dir "${OUTPUT_DIR}"
done

# ---- Performance summary -------------------------------------------------
# Compact per-(dataset, ratio) table read back from the saved JSON: wall-clock
# and total-token throughput.
echo ""
echo "===== performance summary: ${SCORER} ${SELECTION} ====="
"$PYTHON" - "${OUTPUT_DIR}" <<'PY'
import json, os, sys
root = sys.argv[1]
rows = {}      # dataset -> {ratio: (elapsed_sec, total_tok_throughput)}
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
        b = d.get("benchmark", {})
        rows.setdefault(ds, {})[r] = (
            b.get("elapsed_sec", d.get("generation_time_sec")),
            b.get("total_token_throughput_tok_per_s"),
        )
        ratios.add(r)
if not rows:
    print("(no results found under", root, ")")
    sys.exit(0)
ratios = sorted(ratios)
w = max(len(d) for d in rows)
hdr = "  ".join(f"r{r:<14}" for r in ratios)
print(f"{'dataset':<{w}}  {hdr}")
print(f"{'':<{w}}  " + "  ".join(f"{'sec / tok/s':<15}" for _ in ratios))
for ds in sorted(rows):
    cells = []
    for r in ratios:
        v = rows[ds].get(r)
        if v is None or v[0] is None:
            cells.append(f"{'--':<15}")
        else:
            sec, tput = v
            tput_s = f"{tput:.0f}" if tput is not None else "--"
            cells.append(f"{sec:6.1f} / {tput_s:<6}")
    print(f"{ds:<{w}}  " + "  ".join(cells))
PY
echo "ALL_DONE"
