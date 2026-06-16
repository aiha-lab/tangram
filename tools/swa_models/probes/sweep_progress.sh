#!/usr/bin/env bash
# Quick progress snapshot for a benchmark_scbench sweep launched in the
# background. tqdm draws its bar with carriage returns, so a raw `tail` shows
# stale text; this rewrites CR→LF and prints the *latest* bar plus completed
# scores, failures, and live GPU use.
#
# Usage:  bash sweep_progress.sh [logfile]
#   logfile defaults to the gemma max-num-seqs=8 resume log.
set -uo pipefail
LOG="${1:-/tmp/swa_m18_5_gemma_sweep/run_maxseqs8.log}"
GPU="${GPU:-2}"

if [[ ! -f "$LOG" ]]; then echo "no log at $LOG"; exit 1; fi

echo "=== log: $LOG ==="
echo "--- completed (dataset ratio → avg score) ---"
grep -E "===== |avg score:|FAILED|sweep done" "$LOG" | sed 's/  \[/ [/' | tail -16
echo "--- live progress bar (current ratio) ---"
tr '\r' '\n' < "$LOG" | grep "Processed prompts" | tail -1
echo "--- GPU $GPU ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader -i "$GPU" 2>/dev/null
echo "--- now: $(date '+%H:%M:%S') ---"
