#!/usr/bin/env bash
# Reproduce the KV-cache compression speedup (no-compression vs ratio=0.3) on a
# single 80GB GPU, for gemma-3-12b-it, gpt-oss-20b, and Qwen2.5-7B-Instruct-1M.
#
# WHAT IS MEASURED
#   The same long-context, concurrent workload is run twice with an IDENTICAL
#   engine configuration; the ONLY thing that differs between the two runs is
#   --ratio (1.0 = no compression, 0.3 = FastKVzip compression + sliding-window
#   eviction). Wall-clock generation time is compared. Any difference is
#   therefore attributable solely to compression.
#
# WHY THERE IS A SPEEDUP (and why it is a fair comparison)
#   At a fixed GPU memory budget the KV cache holds a fixed number of tokens.
#   With no compression, a few long-context requests already fill the pool, so
#   the scheduler must PREEMPT and re-prefill (recompute) requests — wasted work
#   that inflates wall-clock. Compression shrinks each request's resident KV, so
#   the same requests fit concurrently and run without preemption. This is a
#   real serving scenario (many concurrent long-context requests on one GPU).
#
#   Fairness safeguards, verifiable by a third party:
#     * Identical config for both ratios — only --ratio changes. Same model,
#       dataset, --num, --max-num-seqs, --max-tokens, --gpu-memory-utilization,
#       --page-group-size and head-group cluster map.
#     * Standard greedy decoding (--temperature 0), deterministic + comparable.
#     * Both ratios produce the SAME number of output tokens (same work).
#     * The report prints, for both ratios, the peak GPU KV-cache usage and the
#       cumulative PREEMPTION count, so one can SEE that the baseline is
#       genuinely KV-limited (preemptions > 0) and compression is not
#       (preemptions == 0) — the baseline is not artificially handicapped.
#
#   Honest scope: the speedup is specific to the KV-limited regime (enough
#   concurrent long-context requests to saturate the pool). At low concurrency,
#   where the baseline already fits, the speedup is ~1x. The per-model --num and
#   memory budget below are chosen to sit just past each model's saturation
#   point — that point differs per model because per-token KV differs ~8x
#   (gemma 384 KB/token, gpt-oss 48 KB/token, qwen 56 KB/token); this is an
#   intrinsic model property, not a tuning trick.
#
# USAGE
#   bash run_compression_speedup.sh <gemma|gptoss|qwen>
#   GPU pinned to physical device 2 (override CUDA_VISIBLE_DEVICES).
set -uo pipefail

REPO=/workspace/vllm-asp
PY="$REPO/benchmarks/tangram/benchmark_scbench.py"
MAPS="$REPO/tools/head_group_clustering/cluster_maps"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="$REPO"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

KEY="${1:?usage: run_compression_speedup.sh <gemma|gptoss|qwen>}"

# Per-model preset. Within a model the two ratio runs share ALL of these, so
# the comparison is fair; across models --num / --gpu-memory-utilization differ
# because the KV-saturation point differs with per-token KV size (see header).
case "$KEY" in
  gemma)
    MODEL=/raid/LLM/gemma-3-12b-it;            PG=4
    CMAP="$MAPS/gemma-3-12b-it_r0.3_pg4.npz";   GATE=fastkvzip
    DATASET=scbench_repoqa; MML=98304; NUM=3;  MNS=3; MAX_TOKENS=128; GPU_MEM=0.90 ;;
  gptoss)
    MODEL=/raid/LLM/gpt-oss-20b;               PG=2
    CMAP="$MAPS/gpt-oss-20b_r0.3_pg2.npz"
    # gpt-oss gate is a local fork checkpoint (not on the HuggingFace Hub, so
    # the "fastkvzip" sentinel 404s); point at the absolute .pt path.
    GATE=/workspace/tangram_impl/FastKVzip-gpt-oss/result_gate/gpt-oss-20b/q8_dim16_sink16.pt
    DATASET=scbench_repoqa; MML=98304; NUM=8;  MNS=8; MAX_TOKENS=128; GPU_MEM=0.45 ;;
  qwen)
    MODEL=/raid/LLM/Qwen2.5-7B-Instruct-1M;    PG=4
    CMAP="$MAPS/qwen25-7b-1m_r0.3_pg4.npz";     GATE=fastkvzip
    DATASET=scbench_repoqa; MML=98304; NUM=8;  MNS=8; MAX_TOKENS=128; GPU_MEM=0.40 ;;
  *) echo "unknown model key '$KEY' (use gemma|gptoss|qwen)"; exit 2 ;;
esac
# Per-run overrides (calibration): set any of these in the env to retune.
NUM="${NUM_OVERRIDE:-$NUM}"; MNS="${MNS_OVERRIDE:-$MNS}"; GPU_MEM="${GPU_MEM_OVERRIDE:-$GPU_MEM}"
PG="${PG_OVERRIDE:-$PG}"; CMAP="${CMAP_OVERRIDE:-$CMAP}"
DATASET="${DATASET_OVERRIDE:-$DATASET}"; MML="${MML_OVERRIDE:-$MML}"
MAX_TOKENS="${MAX_TOKENS_OVERRIDE:-$MAX_TOKENS}"
# Compression chunk size is identical for both ratios (it is inert at ratio=1.0,
# which runs no compression), so adjusting it keeps the "only --ratio differs"
# fairness invariant. A smaller chunk lowers the in-flight prefill peak, which
# matters for dense models whose per-request KV is otherwise close between the
# two ratios.
CHUNK="${CHUNK_OVERRIDE:-8192}"

OUT_BASE="${OUT_DIR:-/tmp/speedup_$KEY}"

run_one () {  # $1 = ratio
    local ratio="$1" od="$OUT_BASE/r${1/./}"
    mkdir -p "$od"
    echo "===== $KEY  ratio=$ratio  num=$NUM mns=$MNS gpu_mem=$GPU_MEM (identical config; only --ratio differs) ====="
    python3 "$PY" \
        -d "$DATASET" --num "$NUM" --ratio "$ratio" \
        --page-group-size "$PG" --head-group-cluster-map "$CMAP" \
        --max-num-seqs "$MNS" --single-turn --max-tokens "$MAX_TOKENS" \
        --force-exact-tokens \
        --temperature 0.0 --enable-log-stats \
        --compression-chunk-size "$CHUNK" --compression-window-size 4096 \
        --compression-n-sink-tokens 32 --compression-floor-min 0 \
        --compression-gate-path "$GATE" \
        -m "$MODEL" --max-model-len "$MML" \
        --gpu-memory-utilization "$GPU_MEM" --output-dir "$od" \
        > "$od/run.log" 2>&1
    echo "  ratio=$ratio exit=$?"
}

run_one 1.0
run_one 0.3

# --- Fair comparison report ---------------------------------------------------
python3 - "$OUT_BASE" "$KEY" <<'PYEOF'
import glob, json, re, sys
base, key = sys.argv[1], sys.argv[2]
def metrics(ratio):
    tag = f"r{ratio.replace('.','')}"
    js = glob.glob(f"{base}/{tag}/*/*.json")
    log = f"{base}/{tag}/run.log"
    d = json.load(open(js[0])) if js else None
    txt = open(log, errors="ignore").read() if glob.glob(log) else ""
    kv = [float(x) for x in re.findall(r"GPU KV cache usage: ([0-9.]+)%", txt)]
    pre = [int(x) for x in re.findall(r"Preemptions: (\d+)", txt)]
    b = d["benchmark"] if d else {}
    return dict(t=b.get("elapsed_sec"), out=b.get("total_output_tokens"),
                kv=max(kv) if kv else None, pre=max(pre) if pre else 0)
b, c = metrics("1.0"), metrics("0.3")
print(f"\n================ {key}: compression speedup (fair: only --ratio differs) ================")
print(f"  {'':18s} {'gen_time(s)':>12s} {'out_tokens':>11s} {'peak_KV%':>9s} {'preemptions':>12s}")
for name, m in (("ratio=1.0 (none)", b), ("ratio=0.3 (compress)", c)):
    print(f"  {name:18s} {m['t']:>12.1f} {str(m['out']):>11s} {str(m['kv']):>9s} {m['pre']:>12d}")
if b['t'] and c['t']:
    print(f"\n  >>> SPEEDUP (r1.0 / r0.3) = {b['t']/c['t']:.2f}x"
          f"   | same output tokens: {b['out']==c['out']}"
          f"   | baseline preempted: {b['pre']>0}, compressed preempted: {c['pre']>0}")
PYEOF
echo "===== done: $KEY (results under $OUT_BASE) ====="
