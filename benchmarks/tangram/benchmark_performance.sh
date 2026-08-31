#!/usr/bin/env bash
# SCBench throughput / latency for one compression method. Same knobs as
# benchmark_scbench.sh, performance protocol instead of accuracy: single-turn,
# every request emitting exactly MAX_TOKENS, so decode work is identical across
# settings and only engine cost varies. ratio=0 is the uncompressed reference.
#
#   SCORER  = fastkvzip | snapkv | keydiff | streamingllm | tova | expected_attention
#   SCOPE   = layer (default) | global | uniform      # axis 1, see budget_scope.py
#
# Results land in performance_results/<scorer>_<selection>/, mirroring
# results_accuracy/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# The driver touches CUDA, and a forked engine core cannot re-initialize it.
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
SCOPE=${SCOPE:-layer}

# ---- Sweep ---------------------------------------------------------------
DATASET=${DATASET:-scbench_repoqa}
RATIOS=${RATIOS:-"0.0 0.7"}
NUM=${NUM:-10}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_TOKENS=${MAX_TOKENS:-512}
# Raise it when one full-length request does not fit: gemma-3-12b at 124k needs
# ~46 GiB of KV, just over what 0.90 leaves.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}

# ---- Args ----------------------------------------------------------------
SELECTION="${SCOPE}"
# A cluster map must match the page group it was built for.
PAGE_GROUP_SIZE=${PAGE_GROUP_SIZE:-4}
METHOD_ARGS=(--compression-scorer "${SCORER}"
             --compression-budget-scope "${SCOPE}")

# Append "<flag> <value>" only when the value is set. An unknown SCORER needs no
# check here — argparse rejects it against the same list, and CacheConfig
# against the scorer registry.
opt() { if [ -n "${2:-}" ]; then METHOD_ARGS+=("$1" "$2"); fi; }

# fastkvzip only; unset lets the engine auto-resolve the gate.
opt --compression-gate-path "${GATE_PATH:-}"
opt --compression-scorer-options "${SCORER_OPTIONS:-}"
# A .npz from tools/head_group_clustering; unset or missing falls back to
# identity (adjacent-head) grouping.
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
# Wall-clock and total-token throughput, read back from the saved JSON.
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
