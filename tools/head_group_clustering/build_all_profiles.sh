#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Regenerate every head-group retention profile from the live vLLM engine
# (build_profile.py), 8 GPUs in parallel. Each profile is one (model, scorer,
# selection level); maps are then built from these by build_cluster_map.py.
#
# Matrix: 4 TP=1 models x 4 scorers x 2 selection levels = 32 profiles.
#   models : qwen3-4b, llama-3.1-8b, gemma-3-12b, gpt-oss-20b
#            (qwen3-30b is excluded: it runs TP=2 at serving time, so its
#             per-rank head layout needs a TP-aware profile -- separate work.)
#   scorers: fastkvzip, keydiff, snapkv, expected_attention
#   levels : crosslayer_head -> profile.npz          (pairs cross-layer maps)
#            perlayer_head    -> profile_perlayer.npz (pairs per-layer maps)
#
# All scorers use ONE unified measurement regime: effectively one-shot. The
# context fits in a single compression chunk (prefill_chunk 32768 >= max input
# 30000+sink), so every scorer sees the whole context at once -- the same regime
# the RULER 8192 accuracy evaluation runs in (its ~8k input is one 8192 chunk),
# and the regime ExpectedAttention's kvpress reference assumes. Profiling all
# scorers identically keeps the profile faithful to evaluation.
#
# Concurrency: 8 GPUs, one profile build pinned per GPU, refilled as each frees.
# Per-GPU VLLM_PORT and compile-cache dirs avoid cross-process races; launches
# are staggered to desync model download / kernel JIT.
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CMAPS="$ROOT/tools/head_group_clustering/cluster_maps"
LOGDIR="${LOGDIR:-/tmp/profile_build_logs}"
# Home filesystem is exec-enabled (unlike a noexec /tmp), so JIT-compiled
# kernels can be mmap'd back in.
CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache/profile_build_cache}"
NGPU="${NGPU:-8}"
STAGGER_SEC="${STAGGER_SEC:-20}"
mkdir -p "$LOGDIR" "$CACHE_ROOT"

declare -A HF=(
  [qwen3-4b-instruct-2507]="Qwen/Qwen3-4B-Instruct-2507"
  [llama-3.1-8b-instruct]="meta-llama/Llama-3.1-8B-Instruct"
  [gemma-3-12b-it]="google/gemma-3-12b-it"
  [gpt-oss-20b]="openai/gpt-oss-20b"
)
MODELS=(qwen3-4b-instruct-2507 llama-3.1-8b-instruct gemma-3-12b-it gpt-oss-20b)
SCORERS=(fastkvzip keydiff snapkv expected_attention)
LEVELS=(crosslayer_head perlayer_head)

# fastkvzip gate-path override (only used by --scorer fastkvzip). The gate
# auto-resolver derives the hmkim97/tangram-gate subfolder from the HF model id
# (e.g. meta-llama/Llama-3.1-8B-Instruct -> "llama-3.1-8b-instruct/..."), which
# matches the upload layout for every model EXCEPT llama: its gate was uploaded
# under "llama3.1-8b-instruct" (no dash, the old local checkout basename). Pin
# the exact Hub path so auto-resolve does not 404. Other models: empty = auto.
declare -A GATE_OVERRIDE=(
  [llama-3.1-8b-instruct]="llama3.1-8b-instruct/q4_dim16_sink16.pt"
)

# Unified one-shot measurement settings (identical for every scorer).
COMMON_ARGS=(--base-ratios 0.3
             --dataset-mix "scbench_kv_short:16,scbench_many_shot:12"
             --prefill-chunk 32768 --window-size 4096 --max-ctx-len 30000)

# Build the job list scorer->level->model so consecutive jobs hit different
# models: each 8-wide wave then mixes models (at most ~2 gpt-oss builds compile
# their MoE kernels at once).
JOBS=()
for s in "${SCORERS[@]}"; do
  for lv in "${LEVELS[@]}"; do
    for m in "${MODELS[@]}"; do
      JOBS+=("$s|$m|$lv")
    done
  done
done

run_job() {
  local spec="$1" gpu="$2"
  local scorer model level
  IFS='|' read -r scorer model level <<< "$spec"
  local hf="${HF[$model]}"
  local mdir="$scorer"; [ "$scorer" = "expected_attention" ] && mdir="ea"
  # Output filename suffix follows the clustering SCOPE (per-layer profile), not
  # the level string; the per-layer-threshold level (perlayer_head) is profiled
  # into the within-layer profile that the per_layer-scope map is built from.
  local suffix=""; [ "$level" = "perlayer_head" ] && suffix="_perlayer"
  local out="$CMAPS/$mdir/$model/profile${suffix}.npz"
  local log="$LOGDIR/${mdir}__${model}__${level}.log"
  local jc="$CACHE_ROOT/gpu$gpu"
  mkdir -p "$jc"
  # fastkvzip: apply the per-model gate override when one is set (else auto).
  local gate_args=()
  if [ "$scorer" = "fastkvzip" ] && [ -n "${GATE_OVERRIDE[$model]:-}" ]; then
    gate_args=(--gate-path "${GATE_OVERRIDE[$model]}")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" \
    VLLM_PORT="$((40000 + gpu * 128))" \
    TORCHINDUCTOR_CACHE_DIR="$jc/inductor" TRITON_CACHE_DIR="$jc/triton" \
    PYTHONPATH="$ROOT" \
    python -m tools.head_group_clustering.build_profile \
      --model "$hf" --scorer "$scorer" --level "$level" \
      "${gate_args[@]}" "${COMMON_ARGS[@]}" --out "$out" > "$log" 2>&1
}

free_gpus=(); for ((g = 0; g < NGPU; g++)); do free_gpus+=("$g"); done
declare -A pid2gpu
ji=0; total=${#JOBS[@]}
echo "[sched] $total jobs over $NGPU GPUs; logs in $LOGDIR"
while [ $ji -lt $total ] || [ ${#pid2gpu[@]} -gt 0 ]; do
  while [ ${#free_gpus[@]} -gt 0 ] && [ $ji -lt $total ]; do
    gpu=${free_gpus[0]}; free_gpus=("${free_gpus[@]:1}")
    spec="${JOBS[$ji]}"; ji=$((ji + 1))
    run_job "$spec" "$gpu" &
    pid2gpu[$!]=$gpu
    echo "[launch $ji/$total] gpu=$gpu $spec (pid $!)"
    sleep "$STAGGER_SEC"
  done
  [ ${#pid2gpu[@]} -eq 0 ] && continue
  wait -n
  for pid in "${!pid2gpu[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      g=${pid2gpu[$pid]}; wait "$pid"; rc=$?
      free_gpus+=("$g"); unset 'pid2gpu[$pid]'
      echo "[done] gpu=$g pid=$pid rc=$rc"
    fi
  done
done
echo "[sched] ALL DONE"
