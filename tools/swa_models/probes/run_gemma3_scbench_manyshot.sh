#!/usr/bin/env bash
# M18.3 diagnostic: tangram-asp gemma-3-12b-it on SCBench scbench_many_shot.
# Faster than scbench_kv_short (gen length 96 vs 512, n=54). many_shot is
# nearly flat under compression in the baseline (FastKVzip-fork gemma-3:
# full 51.85 / r0.7 51.48 / r0.5 50.37 / r0.3 50.00 — relative ~96% at r0.3),
# so it is a good catastrophic-loss detector: if tangram stays flat, the
# compression preserves info (kv_short's low absolute is task/harness); if
# tangram collapses, the compression path is broken.
#
# Ratios ordered 1.0 (ceiling, also the harness-confound check) → 0.3 (max
# compression) → 0.5 so the two decisive endpoints land first.
# NOTE: many_shot metric is format-collapse prone / non-monotonic; use mainly
# for WITHIN-tangram relative degradation, not cross-stack absolute equality.
#
# Invoke directly. GPU pinned to physical device 3.
set -euo pipefail

REPO=/workspace/vllm-asp
PY="$REPO/benchmarks/tangram/benchmark_scbench.py"
OUT_DIR="${OUT_DIR:-/tmp/swa_m18_3_ms}"
mkdir -p "$OUT_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONPATH="$REPO"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

MODEL=/raid/LLM/gemma-3-12b-it
DATASET=scbench_many_shot
NUM=${NUM:-54}
RATIOS=(1.0 0.3 0.5)

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
        --single-turn \
        --gpu-memory-utilization 0.70 \
        --output-dir "$OUT_DIR"
done

echo "===== done; results under $OUT_DIR/$DATASET ====="
