#!/usr/bin/env bash
# M18.5 accuracy gate: tangram-asp gpt-oss-20b compression accuracy on SCBench
# scbench_kv_short, compared to the baseline FastKVzip-gpt-oss reference.
#
# Baseline FastKVzip-gpt-oss avg_score (100 samples, chunk8k/w4096, scored by
# FastKVzip-gpt-oss/prefill/results/parse.py -d scbench_kv_short --task qa):
#   ratio 1.0 = 85.40 | 0.7 = 86.70 | 0.5 = 86.40 | 0.3 = 44.30
#
# Baseline-equivalent config: chunk=8192, window=4096, n_sink=32, floor_min=0
# (floor disabled = baseline-faithful). Compression applies only to the 12
# full-attention layers (odd indices); sliding-window layers keep full KV. The
# per-head attention sink is recovered by the FlashAttention log-sum-exp
# correction (M18.4). gpt-oss's gate is a local checkpoint (not on HF Hub), so
# the explicit path is passed rather than the "fastkvzip" sentinel.
#
# Fair-accuracy protocol (see 18-swa-models/HANDOFF §2.4): natural multi-turn
# generation with EOS — NO --single-turn / --force-exact-tokens (those are
# preemption-forcing probes that are not comparable to the baseline).
#
# Invoke directly (do not wrap). GPU defaults to physical device 2; override
# with CUDA_VISIBLE_DEVICES. NUM / RATIOS overridable via env.
set -euo pipefail

REPO=/workspace/vllm-asp
PY="$REPO/benchmarks/tangram/benchmark_scbench.py"
OUT_DIR="${OUT_DIR:-/tmp/swa_m18_5}"
mkdir -p "$OUT_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="$REPO"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

MODEL=/raid/LLM/gpt-oss-20b
GATE=/workspace/tangram_impl/FastKVzip-gpt-oss/result_gate/gpt-oss-20b/q8_dim16_sink16.pt
DATASET=scbench_kv_short
NUM=${NUM:-100}
read -r -a RATIOS <<< "${RATIOS:-0.3 0.5 0.7 1.0}"

# Optional head-group cluster map (same-memory comparison vs identity grouping).
# Unset = identity adjacent-head grouping (over-allocates but needs no map).
CLUSTER_MAP="${CLUSTER_MAP:-}"
map_args=()
if [[ -n "$CLUSTER_MAP" ]]; then
    map_args=(--head-group-cluster-map "$CLUSTER_MAP")
    echo "Using head-group cluster map: $CLUSTER_MAP"
fi

for RATIO in "${RATIOS[@]}"; do
    echo "===== gpt-oss ${DATASET} ratio=${RATIO} (num=${NUM}) ====="
    python3 "$PY" \
        -d "$DATASET" \
        --num "$NUM" \
        --ratio "$RATIO" \
        --page-group-size 2 \
        --compression-chunk-size 8192 \
        --compression-window-size 4096 \
        --compression-n-sink-tokens 32 \
        --compression-floor-min 0 \
        --compression-gate-path "$GATE" \
        "${map_args[@]}" \
        -m "$MODEL" \
        --max-model-len 32768 \
        --max-tokens 96 \
        --gpu-memory-utilization 0.70 \
        --output-dir "$OUT_DIR"
done

echo "===== done; results under $OUT_DIR/$DATASET ====="
