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
# answer-length variance. ratio=1.0 is the uncompressed reference; uniform vs
# non-uniform differ only at ratio<1.0.
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
# Selection level (axis 1): crosslayer_head | perlayer_head | crosslayer_cluster
# | perlayer_cluster | uniform.
LEVEL=${LEVEL:-crosslayer_head}

# ---- Sweep ---------------------------------------------------------------
DATASET=${DATASET:-scbench_repoqa}
RATIOS=${RATIOS:-"1.0 0.3"}
NUM=${NUM:-10}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_TOKENS=${MAX_TOKENS:-512}

# Page-group size is 4 for every method; PAGE_GROUP_SIZE overrides it (the
# fastkvzip cluster map below must then match the chosen page group).
PAGE_GROUP_SIZE=${PAGE_GROUP_SIZE:-4}

# ---- Method-specific args ------------------------------------------------
METHOD_ARGS=(--compression-scorer "${SCORER}" --compression-level "${LEVEL}")
SELECTION="${LEVEL}"

case "${SCORER}" in
    fastkvzip)
        # Gate-based; the gate auto-resolves from the model. The head-group
        # cluster map (if provided) is applied by the common block below.
        :
        ;;
    snapkv)
        # Gate-free observation-window attention; tunable window / pool kernel.
        METHOD_ARGS+=(--compression-snap-window "${SNAP_WINDOW:-32}"
                      --compression-snap-kernel "${SNAP_KERNEL:-7}")
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
        --ratio "${RATIO}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
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
ratios = sorted(ratios, reverse=True)
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
