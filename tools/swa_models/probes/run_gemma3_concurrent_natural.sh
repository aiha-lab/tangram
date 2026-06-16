#!/usr/bin/env bash
# M18.3 follow-up: does the collapse survive removing ONLY the artificial
# force-exact stress, while keeping preemption pressure (low mem) and natural
# concurrency? Serial+natural already proved the compression path is correct
# (r0.3=0.95). This isolates whether force-exact was load-bearing for the
# collapse, or whether natural concurrent generation under preempt pressure
# also corrupts.
#   - NO --max-num-seqs (default concurrency)
#   - NO force-exact (natural EOS)
#   - multi-turn (all qa-pairs)
#   - mem=0.58 (the recorded preempt-thrash point) + n=10
# Verdict: ~0.9+ => collapse was an artifact of force-exact stress, not a
# real concurrent-compression bug. <<0.9 / degenerate => real concurrency bug.
set -euo pipefail
REPO=/workspace/vllm-asp
PY="$REPO/benchmarks/tangram/benchmark_scbench.py"
OUT_DIR="${OUT_DIR:-/tmp/swa_m18_3_concurrent}"
LOG_DIR="${LOG_DIR:-/tmp/swa_m18_3_concurrent/logs}"
mkdir -p "$OUT_DIR" "$LOG_DIR"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="$REPO"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
MODEL="${MODEL:-/raid/LLM/gemma-3-12b-it}"
DATASET=scbench_kv_short
NUM=${NUM:-10}
MEM=${MEM:-0.58}
MAXLEN=${MAXLEN:-26624}
for RATIO in ${RATIOS:-1.0 0.3}; do
    LOG="$LOG_DIR/conc_r${RATIO}_n${NUM}_mem${MEM}.log"
    echo "===== gemma-3 ${DATASET} r=${RATIO} n=${NUM} mem=${MEM} CONCURRENT NATURAL multi-turn ====="
    python3 "$PY" -d "$DATASET" --num "$NUM" --ratio "$RATIO" \
        --page-group-size 2 --compression-chunk-size 8192 \
        --compression-window-size 4096 --compression-n-sink-tokens 32 \
        --compression-floor-min 0 -m "$MODEL" \
        --max-model-len "$MAXLEN" --max-tokens 512 \
        --gpu-memory-utilization "$MEM" --output-dir "$OUT_DIR" 2>&1 | tee "$LOG"
done
echo "===== done; results under $OUT_DIR/$DATASET ====="
