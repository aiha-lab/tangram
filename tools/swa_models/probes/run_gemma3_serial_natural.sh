#!/usr/bin/env bash
# M18.3 decisive isolation: does gemma-3 compression corruption survive STRICT
# serialization (max_num_seqs=1) with NATURAL generation?
#
# Rationale: the recorded collapse (r0.3 score 0.08) was measured with the
# preempt-forcing probe config (--single-turn --force-exact-tokens), which
# (a) does not match the FastKVzip HF baseline (multi-turn, natural EOS) and
# (b) was explicitly built to trigger preemption. This run removes BOTH
# confounds at once:
#   - max_num_seqs=1  -> one request at a time: no batching, no preemption,
#                        no mixed steps. Strictly serial, like baseline HF.
#   - no force-exact  -> natural EOS, real generation length.
#   - multi-turn      -> all 10 qa-pairs per context, like baseline.
# Compression config (pg=2, chunk8192, w4096, nsink32, floor0) matches the
# broken run, so the ONLY removed variable is concurrency/preemption.
#
# Verdict:
#   r0.3 still degenerate under serial+natural -> per-request STATE bug
#       (compression state leaking across requests); preemption is NOT the cause.
#   r0.3 clean (~baseline 18.9% relative) -> corruption needs concurrency;
#       the implementation's serial path is correct.
#   r1.0 is the no-compression ceiling control.
#
# Invoke directly. GPU pinned to physical device 2 (this session).
set -euo pipefail

REPO=/workspace/vllm-asp
PY="$REPO/benchmarks/tangram/benchmark_scbench.py"
OUT_DIR="${OUT_DIR:-/tmp/swa_m18_3_serial}"
LOG_DIR="${LOG_DIR:-/tmp/swa_m18_3_serial/logs}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="$REPO"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

MODEL="${MODEL:-/raid/LLM/gemma-3-12b-it}"
DATASET=scbench_kv_short
NUM=${NUM:-20}
RATIOS=(${RATIOS:-1.0 0.3})

for RATIO in "${RATIOS[@]}"; do
    LOG="$LOG_DIR/serial_r${RATIO}_n${NUM}.log"
    echo "===== gemma-3 ${DATASET} ratio=${RATIO} n=${NUM} SERIAL(max_num_seqs=1) NATURAL multi-turn ====="
    echo "log -> $LOG"
    python3 "$PY" \
        -d "$DATASET" \
        --num "$NUM" \
        --ratio "$RATIO" \
        --max-num-seqs 1 \
        --page-group-size 2 \
        --compression-chunk-size 8192 \
        --compression-window-size 4096 \
        --compression-n-sink-tokens 32 \
        --compression-floor-min 0 \
        -m "$MODEL" \
        --max-model-len 32768 \
        --max-tokens 512 \
        --gpu-memory-utilization 0.85 \
        --output-dir "$OUT_DIR" 2>&1 | tee "$LOG"
done

echo
echo "===== done; results under $OUT_DIR/$DATASET ====="