#!/usr/bin/env bash
# Compression showcase: KV-capacity-limited concurrency on a long context.
#
# Picks a regime where the no-compression baseline (RATIO=1.0) cannot hold the
# requested concurrency in KV and thrashes (preempt + full re-prefill), while
# compression (RATIO<1.0, with sliding-window eviction) shrinks per-request KV
# enough that all requests fit and run concurrently. The same config is used
# for both ratios; only --ratio differs, so the wall-clock gap is attributable
# to compression.
#
# gemma-3-12b-it, scbench_repoqa (~85k context). At RATIO=1.0 each request holds
# ~70% of the KV pool (no compression, no sliding-window eviction), so 2+ long
# requests cannot coexist; at RATIO=0.3 each holds ~11%, so 4 fit. Single-turn +
# small --max-tokens keeps each sample fast.
#
# Invoke directly. GPU pinned to physical device 2 (override CUDA_VISIBLE_DEVICES).
set -euo pipefail

REPO=/workspace/vllm-asp
PY="$REPO/benchmarks/tangram/benchmark_scbench.py"
OUT_DIR="${OUT_DIR:-/tmp/swa_showcase}"
mkdir -p "$OUT_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="$REPO"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

MODEL="${MODEL:-/raid/LLM/gemma-3-12b-it}"
CLUSTER_MAP="${CLUSTER_MAP:-$REPO/tools/head_group_clustering/cluster_maps/gemma-3-12b-it_r0.3_pg4.npz}"
DATASET="${DATASET:-scbench_repoqa}"
NUM=${NUM:-4}
MNS=${MNS:-4}
RATIO=${RATIO:-0.3}
MML=${MML:-98304}
MAX_TOKENS=${MAX_TOKENS:-256}
GPU_MEM=${GPU_MEM:-0.90}

echo "===== showcase: ${DATASET} ratio=${RATIO} num=${NUM} mns=${MNS} \
max_tokens=${MAX_TOKENS} single-turn (pg4 cluster-map) ====="
python3 "$PY" \
    -d "$DATASET" \
    --num "$NUM" \
    --ratio "$RATIO" \
    --page-group-size 4 \
    --head-group-cluster-map "$CLUSTER_MAP" \
    --max-num-seqs "$MNS" \
    --single-turn \
    --max-tokens "$MAX_TOKENS" \
    --enable-log-stats \
    --compression-chunk-size 8192 \
    --compression-window-size 4096 \
    --compression-n-sink-tokens 32 \
    --compression-floor-min 0 \
    --compression-gate-path fastkvzip \
    -m "$MODEL" \
    --max-model-len "$MML" \
    --gpu-memory-utilization "$GPU_MEM" \
    --output-dir "$OUT_DIR"

echo "===== done; results under $OUT_DIR/$DATASET ====="
