#!/usr/bin/env bash
# M18.3 root-cause: tangram-asp gemma-3-12b-it on scbench_kv_short with config
# matched to the FastKVzip baseline as closely as the vLLM harness allows, to
# isolate whether the kv collapse is a real bug (user's hypothesis) vs a config
# artifact.
#
# Baseline (FastKVzip-fork) reference avg_score: full 28.70 / r0.7 26.30 /
# r0.5 23.20 / r0.3 18.90 (results/parse.py, 100 samples).
#
# Matched to baseline: prefill_chunk=8192, window=4096, max_new_tokens=512,
# NO force-exact (natural EOS, like baseline), floor_min=0 (no floor), and
# page_group_size=1 (per-head eviction, matching baseline's per-head FastKVzip;
# pg>1 max-pools and would keep MORE, not less). n_sink=32 (baseline uses the
# instruction-prefix length start_idx; closest fixed value).
#
# r1.0 first = the ceiling the user asked for (no compression ⇒ config-robust).
# Invoke directly. GPU pinned to physical device 3.
set -euo pipefail

REPO=/workspace/vllm-asp
PY="$REPO/benchmarks/tangram/benchmark_scbench.py"
OUT_DIR="${OUT_DIR:-/tmp/swa_m18_3_match}"
mkdir -p "$OUT_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONPATH="$REPO"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

MODEL=/raid/LLM/gemma-3-12b-it
DATASET=scbench_kv_short
NUM=${NUM:-100}
RATIOS=(1.0 0.3 0.5)

for RATIO in "${RATIOS[@]}"; do
    echo "===== gemma-3 ${DATASET} ratio=${RATIO} (baseline-match, pg=1, no force-exact) ====="
    python3 "$PY" \
        -d "$DATASET" \
        --num "$NUM" \
        --ratio "$RATIO" \
        --page-group-size 1 \
        --compression-chunk-size 8192 \
        --compression-window-size 4096 \
        --compression-n-sink-tokens 32 \
        --compression-floor-min 0 \
        -m "$MODEL" \
        --max-model-len 32768 \
        --max-tokens 512 \
        --single-turn \
        --gpu-memory-utilization 0.70 \
        --output-dir "$OUT_DIR"
done

echo "===== done; results under $OUT_DIR/$DATASET ====="
