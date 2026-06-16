#!/usr/bin/env bash
# Ad-hoc accuracy probe: tangram-asp gemma-3-12b-it on SCBench scbench_repoqa
# with the head-group cluster map applied, ratio=0.3, num=1.
#
# Unlike run_gemma3_scbench_full.sh (identity adjacency at page-group-size 2),
# this applies the real cross-layer cluster map
# tools/head_group_clustering/cluster_maps/gemma-3-12b-it_r0.3_pg4.npz, which is
# built at page_group_size=4 over the 8 full-attention layers
# ([5,11,17,23,29,35,41,47]); the runtime --page-group-size MUST therefore be 4
# to match the map (the loader rejects a mismatch). gemma-3-12b-it has 8 KV
# heads/layer, so pg=4 yields 2 clusters/layer.
#
# Config is otherwise baseline-faithful (chunk=8192, window=4096, n_sink=32,
# floor_min=0, gate=fastkvzip sentinel) and uses the natural multi-turn EOS
# protocol that the full sweep uses for repoqa (no --single-turn /
# --force-exact-tokens). repoqa is scored by F1 similarity (scbench_local).
#
# Invoke directly (do not wrap). GPU pinned to physical device 3.
set -euo pipefail

REPO=/workspace/vllm-asp
PY="$REPO/benchmarks/tangram/benchmark_scbench.py"
OUT_DIR="${OUT_DIR:-/tmp/swa_repoqa_clustermap}"
mkdir -p "$OUT_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONPATH="$REPO"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

MODEL=/raid/LLM/gemma-3-12b-it
CLUSTER_MAP="${CLUSTER_MAP:-$REPO/tools/head_group_clustering/cluster_maps/gemma-3-12b-it_r0.3_pg4.npz}"
DATASET=scbench_repoqa
NUM=${NUM:-1}
RATIO=${RATIO:-0.3}
MML=${MML:-98304}
GPU_MEM=${GPU_MEM:-0.90}

# Optional cluster map: empty CLUSTER_MAP -> identity (adjacent-head) grouping.
map_args=()
[[ -n "$CLUSTER_MAP" ]] && map_args=(--head-group-cluster-map "$CLUSTER_MAP")
# Concurrency cap: empty -> vLLM default (unbounded).
seq_args=()
[[ -n "${MAX_NUM_SEQS:-}" ]] && seq_args=(--max-num-seqs "$MAX_NUM_SEQS")
# Engine stats logger (preemption count) for diagnosing concurrency thrash.
stats_args=()
[[ "${LOG_STATS:-1}" == "1" ]] && stats_args=(--enable-log-stats)

echo "===== gemma-3 ${DATASET} ratio=${RATIO} num=${NUM} pg=4 \
mns=${MAX_NUM_SEQS:-default} map=${CLUSTER_MAP:-identity} ====="
python3 "$PY" \
    -d "$DATASET" \
    --num "$NUM" \
    --ratio "$RATIO" \
    --page-group-size 4 \
    "${map_args[@]}" \
    "${seq_args[@]}" \
    "${stats_args[@]}" \
    --compression-chunk-size 8192 \
    --compression-window-size 4096 \
    --compression-n-sink-tokens 32 \
    --compression-floor-min 0 \
    --compression-gate-path fastkvzip \
    -m "$MODEL" \
    --max-model-len "$MML" \
    --max-tokens 512 \
    --gpu-memory-utilization "$GPU_MEM" \
    --output-dir "$OUT_DIR"

echo "===== done; results under $OUT_DIR/$DATASET ====="
