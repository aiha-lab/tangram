#!/usr/bin/env bash
# M18.3 accuracy gate: tangram-asp gemma-3-12b-it compression accuracy on
# SCBench scbench_kv_short, compared to the baseline FastKVzip-gemma-3 reference
# (baseline avg_score: full 28.70 / r0.7 26.30 / r0.5 23.20 / r0.3 18.90 — from
# /workspace/tangram_impl/fastkvzip-accuracy-reproduce, scored by results/parse.py).
#
# Baseline-equivalent config: chunk=8192, window=4096, n_sink=32, floor_min=0
# (floor disabled = baseline-faithful), NO head-group cluster map (sliding-window
# hybrids do not support clustering yet). Compression applies only to the 8
# full-attention layers (M18.1); sliding-window layers keep full KV.
#
# Invoke directly (do not wrap). GPU pinned to physical device 3.
set -euo pipefail

REPO=/workspace/vllm-asp
PY="$REPO/benchmarks/tangram/benchmark_scbench.py"
OUT_DIR="${OUT_DIR:-/tmp/swa_m18_3}"
mkdir -p "$OUT_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONPATH="$REPO"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

MODEL=/raid/LLM/gemma-3-12b-it
DATASET=scbench_kv_short
NUM=${NUM:-100}
RATIOS=(0.3 0.5 0.7 1.0)

for RATIO in "${RATIOS[@]}"; do
    echo "===== gemma-3 ${DATASET} ratio=${RATIO} ====="
    python3 "$PY" \
        -d "$DATASET" \
        --num "$NUM" \
        --ratio "$RATIO" \
        --page-group-size 2 \
        --compression-chunk-size 8192 \
        --compression-window-size 4096 \
        --compression-n-sink-tokens 32 \
        --compression-floor-min 0 \
        -m "$MODEL" \
        --max-model-len 32768 \
        --max-tokens 512 \
        --single-turn \
        --force-exact-tokens \
        --gpu-memory-utilization 0.70 \
        --output-dir "$OUT_DIR"
done

echo "===== done; results under $OUT_DIR/$DATASET ====="
