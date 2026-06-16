#!/usr/bin/env bash
# M18.5 full SCBench sweep: tangram-asp gpt-oss-20b compression accuracy across
# every SCBench task the baseline FastKVzip-gpt-oss has results for, compared to
# that baseline. Mirrors run_gptoss_scbench_kv.sh's config (chunk8192/w4096/
# sink32/floor0, identity pg=2, natural multi-turn EOS) but iterates datasets.
#
# Per-dataset max-model-len is sized to the measured context length (avoids
# silent truncation): tiny QA 4k, mid-context 32-64k, long-context 131k (the
# gpt-oss limit). Long multi-turn tasks (qa_eng/choice_eng/summary/vt ~120k) sit
# near that limit; they run last and a per-dataset failure is logged and skipped
# rather than aborting the sweep.
#
# Baseline numbers: FastKVzip-gpt-oss/prefill/results/parse.py
#   (per dataset: python -m results.parse -m gpt-oss-20b_fastkvzip_chunk8k_w4096
#    -d <dataset> --task <qa|reason> -n <num>).
#
# Invoke directly. GPU defaults to device 2 (override CUDA_VISIBLE_DEVICES).
# DATASETS env overrides the list; CLUSTER_MAP env adds a head-group map.
set -uo pipefail   # NOTE: no -e — one dataset's failure must not abort the sweep.

REPO=/workspace/vllm-asp
PY="$REPO/benchmarks/tangram/benchmark_scbench.py"
OUT_DIR="${OUT_DIR:-/tmp/swa_m18_5_sweep}"
mkdir -p "$OUT_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="$REPO"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

MODEL=/raid/LLM/gpt-oss-20b
GATE=/workspace/tangram_impl/FastKVzip-gpt-oss/result_gate/gpt-oss-20b/q8_dim16_sink16.pt
read -r -a RATIOS <<< "${RATIOS:-0.3 0.5 0.7 1.0}"

CLUSTER_MAP="${CLUSTER_MAP:-}"
map_args=()
[[ -n "$CLUSTER_MAP" ]] && map_args=(--head-group-cluster-map "$CLUSTER_MAP")

# "dataset:max_model_len:num" — fast/short first, long-context last. num matches
# the baseline's available sample count for an apples-to-apples comparison.
SPECS_DEFAULT=(
    "gsm:4096:100"
    "squad:4096:100"
    "scbench_kv_short:32768:100"
    "scbench_prefix_suffix_short:32768:100"
    "scbench_many_shot:32768:54"
    "scbench_mf_mid:65536:100"
    "scbench_repoqa:131072:88"
    "scbench_summary:131072:70"
    "scbench_vt:131072:90"
    "scbench_qa_eng:131072:20"
    "scbench_choice_eng:131072:18"
)
if [[ -n "${DATASETS:-}" ]]; then
    read -r -a SPECS <<< "$DATASETS"
else
    SPECS=("${SPECS_DEFAULT[@]}")
fi

for spec in "${SPECS[@]}"; do
    IFS=":" read -r DATASET MML NUM <<< "$spec"
    for RATIO in "${RATIOS[@]}"; do
        echo "===== gpt-oss ${DATASET} ratio=${RATIO} (num=${NUM} mml=${MML}) ====="
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
            --max-model-len "$MML" \
            --max-tokens 512 \
            --gpu-memory-utilization 0.80 \
            --output-dir "$OUT_DIR" \
            || echo "!!!!! FAILED: ${DATASET} ratio=${RATIO} (continuing) !!!!!"
    done
done

echo "===== sweep done; results under $OUT_DIR ====="
