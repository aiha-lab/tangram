#!/usr/bin/env bash
# Full SCBench sweep for gemma-3-12b-it — same protocol as
# run_gptoss_scbench_full.sh (identity pg=2, chunk8192/w4096/sink32/floor0,
# natural multi-turn EOS, num = baseline sample count), compared to the baseline
# FastKVzip-gemma-3 reference in
# /workspace/tangram_impl/fastkvzip-accuracy-reproduce/prefill/results.
#
# gemma-3 differences from gpt-oss that matter here:
#   - gate = the "fastkvzip" sentinel (HF auto-resolves to
#     gemma-3-12b-it/q2_dim16_sink16.pt, 8 full-attention static layers).
#   - head_dim 256 (vs gpt-oss 64) ⇒ KV ≈ 384 KB/token (~8× gpt-oss). Full KV for
#     a ~130k-token context is ~49 GB, so per-dataset max-model-len is sized to
#     the gemma-tokenizer context length (not a flat 131072) and the longest
#     tasks (summary/vt/qa_eng/choice_eng) may exceed memory at low compression;
#     a per-dataset failure is logged and skipped (set -e is intentionally off).
#   - gemma's prompt template already exists in scbench_local.template.
#
# Invoke directly. GPU defaults to device 2. DATASETS/CLUSTER_MAP/RATIOS env
# override as in the gpt-oss runner.
set -uo pipefail

REPO=/workspace/vllm-asp
PY="$REPO/benchmarks/tangram/benchmark_scbench.py"
OUT_DIR="${OUT_DIR:-/tmp/swa_m18_5_gemma_sweep}"
mkdir -p "$OUT_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="$REPO"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

MODEL=/raid/LLM/gemma-3-12b-it
GATE="${GATE:-fastkvzip}"        # sentinel → gemma-3-12b-it/q2_dim16_sink16.pt
GPU_MEM="${GPU_MEM:-0.90}"
# Cap concurrent sequences. Long multi-turn tasks (repoqa/summary/...) at high
# concurrency overflow the KV cache and trigger preemption→recompute thrash
# (see memory project_repoqa_cliff); gemma's large per-token KV makes it
# pathological (repoqa NUM=88 took 9h+ then OOM-failed). A small cap bounds KV
# pressure and restores normal throughput. Empty = vLLM default (unbounded).
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
seq_args=()
[[ -n "$MAX_NUM_SEQS" ]] && seq_args=(--max-num-seqs "$MAX_NUM_SEQS")
read -r -a RATIOS <<< "${RATIOS:-0.3 0.5 0.7 1.0}"

CLUSTER_MAP="${CLUSTER_MAP:-}"
map_args=()
[[ -n "$CLUSTER_MAP" ]] && map_args=(--head-group-cluster-map "$CLUSTER_MAP")

# "dataset:max_model_len:num" — max_model_len sized to the gemma-tokenizer
# context (measured: gsm 0.1k, squad 0.5k, kv_short 25k, prefix_suffix 18k,
# many_shot 26k, mf_mid 75k, repoqa 85k, summary 117k, vt 130k, qa_eng 121k,
# choice_eng 126k) plus headroom for the multi-turn questions, capped at 131072.
SPECS_DEFAULT=(
    "gsm:4096:100"
    "squad:4096:100"
    "scbench_kv_short:32768:100"
    "scbench_prefix_suffix_short:32768:100"
    "scbench_many_shot:32768:54"
    "scbench_mf_mid:90112:100"
    "scbench_repoqa:98304:88"
    "scbench_summary:126976:70"
    "scbench_vt:131072:90"
    "scbench_qa_eng:131072:20"
    "scbench_choice_eng:131072:18"
)
if [[ -n "${DATASETS:-}" ]]; then read -r -a SPECS <<< "$DATASETS"; else SPECS=("${SPECS_DEFAULT[@]}"); fi

for spec in "${SPECS[@]}"; do
    IFS=":" read -r DATASET MML NUM <<< "$spec"
    for RATIO in "${RATIOS[@]}"; do
        echo "===== gemma-3 ${DATASET} ratio=${RATIO} (num=${NUM} mml=${MML}) ====="
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
            "${seq_args[@]}" \
            -m "$MODEL" \
            --max-model-len "$MML" \
            --max-tokens 512 \
            --gpu-memory-utilization "$GPU_MEM" \
            --output-dir "$OUT_DIR" \
            || echo "!!!!! FAILED: ${DATASET} ratio=${RATIO} (continuing) !!!!!"
    done
done

echo "===== gemma-3 sweep done; results under $OUT_DIR ====="
