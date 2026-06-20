#!/usr/bin/env bash
# Simplest Tangram demo: ONE RULER-8K sample with snapkv KV-cache compression at
# ratio 0.5. Pick a GPU by prefixing, e.g. `CUDA_VISIBLE_DEVICES=0 ./benchmark_ruler_single.sh`.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# vLLM V1's engine core must spawn (not fork) once CUDA is initialized in the driver.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="../..:${PYTHONPATH:-}"

# window/floor are shrunk from their long-context defaults so the ratio actually
# binds on RULER's short 8K context (otherwise the always-keep window covers it).
python3 benchmark_ruler.py \
    -m Qwen/Qwen3-4B-Instruct-2507 \
    -l 8192 --tasks niah_single_1 --num 100 \
    --ratio 0.5 --compression-scorer snapkv \
    --compression-window-size 32 --compression-floor-min 0 \
