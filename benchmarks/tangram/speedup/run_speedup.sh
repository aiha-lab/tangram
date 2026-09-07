#!/usr/bin/env bash
# E2E speedup of Tangram KV-cache compression: the uncompressed baseline vs a
# compressed run, wall-clock generation time on an SCBench task. Thin shim over
# benchmark_performance.sh (single-turn, exact token budget -> apples-to-apples).
#
# --compression-ratio is the EVICTED fraction, in [0, 1): 0 is the baseline,
# 0.9 evicts 90%. The reference column is the smallest ratio in RATIOS.
#
# Axes, each a space-separated env list:
#   MODELS       qwen3-4b (default) | gemma | gptoss   -- per-model preset below
#   SCORERS      snapkv fastkvzip ...
#   GRAPH_MODES  eager  = enforce_eager, the historical measurement mode
#                graph  = VLLM_COMPILE + piecewise CUDA graphs (TANGRAM_GRAPH=1;
#                         ragged runs are pinned to PIECEWISE at config time)
#
# WHY THE SPEEDUP EXISTS, AND WHY THE COMPARISON IS FAIR
#   At a fixed memory budget the KV cache holds a fixed number of tokens. With
#   no compression a few long-context requests already fill the pool, so the
#   scheduler must preempt and re-prefill -- wasted work that inflates
#   wall-clock. Compression shrinks each request's resident KV, so the same
#   requests fit concurrently and run without preemption.
#
#   Within one row every knob except --compression-ratio is identical, and
#   --force-exact-tokens makes both ratios emit the same number of output
#   tokens, so the decode work is equal and only engine cost varies. The
#   FAIRNESS table then prints each run's peak KV occupancy and cumulative
#   preemption count, so a reader can see that the baseline is genuinely
#   KV-limited (preemptions > 0) and the compressed run is not (preemptions
#   == 0) -- the baseline is not handicapped to manufacture the gap.
#
# The sliding-window eviction demo (gemma-3 under KV-limited concurrency) is
# this script with one preset selected:
#   MODELS=gemma SCORERS=fastkvzip RATIOS="0.0 0.7" GRAPH_MODES=eager \
#       bash run_speedup.sh
#
# NOTE on the baseline: it runs WITH ragged paging (PAGE_GROUP_SIZE, default 4),
# so the reported speedup is compression vs uncompressed-Tangram -- NOT vs
# vanilla vLLM. Vanilla runs page_group_size=None and is substantially faster at
# long context (measured 3.08x vs pg=4 uncompressed at ~125k ctx); use it when
# comparing against upstream, not this script's reference column.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # benchmarks/tangram
REPO_ROOT="$(cd "${DIR}/../.." && pwd)"

GPU_ID=${GPU_ID:-0}
MODELS=${MODELS:-qwen3-4b}
SCORERS=${SCORERS:-"snapkv fastkvzip"}   # one run per scorer
SCOPE=${SCOPE:-layer}
RATIOS=${RATIOS:-"0.0 0.75 0.9"}
GRAPH_MODES=${GRAPH_MODES:-"eager graph"}   # subset to skip a mode, e.g. "graph"
# The peak-KV and preemption columns come from the engine's periodic stats
# logger, which benchmark_performance.sh captures per ratio. Turning this off
# leaves the fairness table blank; the speedup tables are unaffected.
LOG_STATS=${LOG_STATS:-1}

# ---- Per-model presets ---------------------------------------------------
# NUM, MAX_NUM_SEQS and the memory budget sit just past each model's KV
# saturation point. That point is a model property, not a tuning trick: they
# differ because per-token KV differs ~8x (gemma 384 KB/token, gpt-oss 48,
# qwen 56). At low concurrency, where the baseline already fits, the speedup is
# ~1x for every model. The dataset belongs to the preset because the saturation
# point is a function of context length.
#
# The gemma and gpt-oss presets were calibrated with FLOOR_MIN=0 (a weak head
# group may be emptied); the engine default floor is 512, which compresses less.
# Set FLOOR_MIN=0 to reproduce the calibration run.
#
# Any of MODEL / DATASET / MAX_LEN / NUM / MAX_NUM_SEQS / GPU_MEM_UTIL /
# PAGE_GROUP_SIZE / MAX_TOKENS set in the environment overrides the preset.
load_preset() {
    case "$1" in
        qwen3-4b)
            P_MODEL=Qwen/Qwen3-4B-Instruct-2507; P_DATASET=scbench_vt
            P_MML=200000; P_NUM=10; P_MNS=16; P_MEM=0.90; P_PG=4; P_TOK=96 ;;
        gemma)
            # Sliding-window hybrid. At ~85k context and no compression each
            # request holds ~70% of the KV pool, so two cannot coexist; at
            # ratio 0.7 each holds ~11% and four fit. NUM=MNS=4 therefore puts
            # the baseline past saturation and the compressed run inside it.
            P_MODEL=google/gemma-3-12b-it;       P_DATASET=scbench_repoqa
            P_MML=98304;  P_NUM=4;  P_MNS=4;  P_MEM=0.90; P_PG=4; P_TOK=128 ;;
        gptoss)
            P_MODEL=openai/gpt-oss-20b;          P_DATASET=scbench_repoqa
            P_MML=98304;  P_NUM=8;  P_MNS=8;  P_MEM=0.45; P_PG=4; P_TOK=128 ;;
        *)  echo "Unknown model key '$1' in MODELS (use qwen3-4b|gemma|gptoss)" >&2
            exit 2 ;;
    esac

    P_MODEL=${MODEL:-$P_MODEL};   P_DATASET=${DATASET:-$P_DATASET}
    P_MML=${MAX_LEN:-$P_MML};     P_NUM=${NUM:-$P_NUM}
    P_MNS=${MAX_NUM_SEQS:-$P_MNS};   P_MEM=${GPU_MEM_UTIL:-$P_MEM}
    P_PG=${PAGE_GROUP_SIZE:-$P_PG};  P_TOK=${MAX_TOKENS:-$P_TOK}
}

# ---- What this does ------------------------------------------------------
cat <<EOF
==========================================================================
 Tangram speedup quick-reproduce
 Measures e2e generation-time speedup of KV-cache compression:
   speedup = time(baseline) / time(compressed), within one execution mode
 and the execution-mode gain: time(eager) / time(graph) per ratio.
 Models  : ${MODELS}
 Scorers : ${SCORERS}
 Modes   : ${GRAPH_MODES}
 Scope   : ${SCOPE}   Ratios: ${RATIOS}   GPU: ${GPU_ID}
==========================================================================
EOF

# The threshold scopes need a head-group map; resolve it from the in-repo
# collection, keyed by scorer + model basename so it follows the model. The
# 'layer' scope needs the per-layer map variant; 'global' uses the cross-layer
# map.
MAP_SUFFIX=""; [ "${SCOPE}" = "layer" ] && MAP_SUFFIX="_perlayer"

BASE="${OUTPUT_DIR:-${DIR}/performance_results}"
# One line per engine run. The summary reads paths from here instead of
# re-deriving them, so a per-model dataset cannot desynchronize the two.
MANIFEST="${BASE}/runs.tsv"
mkdir -p "${BASE}"
: > "${MANIFEST}"

for MKEY in ${MODELS}; do
    load_preset "${MKEY}"
    MODEL_KEY=$(basename "${P_MODEL}" | tr 'A-Z' 'a-z')

    for SCORER in ${SCORERS}; do
        MAP="${REPO_ROOT}/tools/head_group_clustering/cluster_maps/${SCORER}/${MODEL_KEY}/pg${P_PG}_r0.3${MAP_SUFFIX}.npz"
        [ -f "${MAP}" ] || { echo "Cluster map not found: ${MAP}" >&2; exit 1; }

        for MODE in ${GRAPH_MODES}; do
            case "${MODE}" in
                eager) TANGRAM_GRAPH_VALUE=0 ;;
                graph) TANGRAM_GRAPH_VALUE=1 ;;
                *) echo "Unknown mode '${MODE}' in GRAPH_MODES (use eager|graph)" >&2
                   exit 1 ;;
            esac

            OUT="${BASE}/${MODEL_KEY}_${SCORER}_${SCOPE}_${MODE}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${MODEL_KEY}" "${SCORER}" "${MODE}" "${P_DATASET}" "${P_PG}" \
                "${OUT}" >> "${MANIFEST}"

            echo ""
            echo ">>> model=${MODEL_KEY}  scorer=${SCORER}  scope=${SCOPE}  mode=${MODE}"
            TANGRAM_GRAPH="${TANGRAM_GRAPH_VALUE}" \
            GPU_ID="${GPU_ID}" MODEL="${P_MODEL}" MAX_LEN="${P_MML}" \
            SCORER="${SCORER}" SCOPE="${SCOPE}" DATASET="${P_DATASET}" \
            RATIOS="${RATIOS}" PAGE_GROUP_SIZE="${P_PG}" \
            MAX_TOKENS="${P_TOK}" NUM="${P_NUM}" MAX_NUM_SEQS="${P_MNS}" \
            GPU_MEM_UTIL="${P_MEM}" LOG_STATS="${LOG_STATS}" \
            HEAD_GROUP_CLUSTER_MAP="${MAP}" OUTPUT_DIR="${OUT}" \
                bash "${DIR}/benchmark_performance.sh" >/dev/null
            echo "    done."
        done
    done
done

# ---- Consolidated summary ------------------------------------------------
# Three tables: (1) per run, per-ratio wall-clock + compression speedup vs the
# baseline ratio; (2) per run, the graph-vs-eager gain; (3) the fairness
# evidence -- peak KV occupancy and cumulative preemptions per ratio.
SCOPE="${SCOPE}" RATIOS="${RATIOS}" GRAPH_MODES="${GRAPH_MODES}" \
    python3 - "${MANIFEST}" <<'PY'
import glob, json, os, re, sys

manifest = sys.argv[1]
scope = os.environ["SCOPE"]
modes = os.environ["GRAPH_MODES"].split()
ratios = [float(r) for r in os.environ["RATIOS"].split()]
ref = min(ratios)                                  # the uncompressed baseline
comp = [r for r in ratios if r != ref]

# (model_key, scorer, mode) -> (dataset, page_group_size, output_dir)
runs = {}
with open(manifest) as f:
    for line in f:
        model, scorer, mode, dataset, pg, out = line.rstrip("\n").split("\t")
        runs[(model, scorer, mode)] = (dataset, pg, out)

if not runs:
    print("(no runs recorded in", manifest, ")")
    sys.exit(0)


def spellings(ratio):
    """How a ratio can appear in a filename. The drivers interpolate the parsed
    float, so 0.0 lands as "0.0"; the %g form is kept because the shell may
    have passed the ratio in that spelling."""
    return (f"{ratio}", f"{ratio:g}")


def result(key, ratio):
    """The saved result JSON for one (run, ratio), or None if it is missing.
    An interrupted sweep leaves a model with only some of its modes, so an
    unrecorded key is a blank cell, not an error."""
    if key not in runs:
        return None
    dataset, pg, out = runs[key]
    d = os.path.join(out, dataset)
    for r in spellings(ratio):
        for path in glob.glob(os.path.join(d, f"*_r{r}_pg{pg}.json")):
            with open(path) as f:
                return json.load(f)
    return None


def elapsed(key, ratio):
    j = result(key, ratio)
    if j is None:
        return None
    b = j.get("benchmark", {})
    return b.get("elapsed_sec", j.get("generation_time_sec"))


def fairness(key, ratio):
    """Peak KV occupancy and total preemptions, read from the captured engine
    log. 'Preemptions: N' is cumulative and the logger omits the field entirely
    while the count is zero, so an absent match means no preemption happened --
    not a parse failure."""
    if key not in runs:
        return None, None
    dataset, _, out = runs[key]
    paths = [os.path.join(out, dataset, f"engine_r{r}.log")
             for r in spellings(ratio)]
    found = next((p for p in paths if os.path.exists(p)), None)
    if found is None:
        return None, None
    with open(found, errors="ignore") as f:
        txt = f.read()
    kv = [float(x) for x in re.findall(r"GPU KV cache usage: ([0-9.]+)%", txt)]
    pre = [int(x) for x in re.findall(r"Preemptions: (\d+)", txt)]
    return (max(kv) if kv else None), (max(pre) if pre else 0)


keys = sorted(runs)
w_run = max(12, max(len(f"{m}/{s}") for m, s, _ in keys))
LINE = "=" * (w_run + 76)


def run_label(key):
    return f"{key[0]}/{key[1]}"


print()
print(LINE)
print(f" SPEEDUP SUMMARY   scope={scope}")
print(f" speedup = time(r={ref:g}) / time(r=compressed), within each mode")
print(LINE)
cols = ([f"r{ref:g} (s)"] + [f"r{r:g} (s)" for r in comp]
        + [f"r{r:g} speedup" for r in comp])
print(f"{'model/scorer':<{w_run}}  {'mode':<6}  " + "  ".join(f"{c:>12}" for c in cols))
for key in keys:
    t_ref = elapsed(key, ref)
    cells = [f"{t_ref:12.1f}" if t_ref else f"{'--':>12}"]
    for r in comp:
        t = elapsed(key, r)
        cells.append(f"{t:12.1f}" if t else f"{'--':>12}")
    for r in comp:
        t = elapsed(key, r)
        sp = (t_ref / t) if (t_ref and t) else None
        cells.append(f"{sp:11.2f}x" if sp else f"{'--':>12}")
    print(f"{run_label(key):<{w_run}}  {key[2]:<6}  " + "  ".join(cells))
print(LINE)

if "eager" in modes and "graph" in modes:
    pairs = sorted({(m, s) for m, s, _ in keys})
    print()
    print(LINE)
    print(" GRAPH-MODE GAIN   gain = time(eager) / time(graph), per ratio")
    print(LINE)
    print(f"{'model/scorer':<{w_run}}  "
          + "  ".join(f"{f'r{r:g}':>12}" for r in ratios))
    for model, scorer in pairs:
        cells = []
        for r in ratios:
            te = elapsed((model, scorer, "eager"), r)
            tg = elapsed((model, scorer, "graph"), r)
            gain = (te / tg) if (te and tg) else None
            cells.append(f"{gain:11.2f}x" if gain else f"{'--':>12}")
        print(f"{model + '/' + scorer:<{w_run}}  " + "  ".join(cells))
    print(LINE)

print()
print(LINE)
print(" FAIRNESS   peak KV occupancy / cumulative preemptions, per ratio")
print(" The speedup is real only if the baseline preempts and the compressed")
print(" run does not; a baseline at 0 preemptions is not KV-limited, and the")
print(" row measures something else.")
print(LINE)
print(f"{'model/scorer':<{w_run}}  {'mode':<6}  "
      + "  ".join(f"{f'r{r:g} KV%/pre':>14}" for r in ratios))
for key in keys:
    cells = []
    for r in ratios:
        kv, pre = fairness(key, r)
        if kv is None and pre is None:
            cells.append(f"{'--':>14}")
            continue
        kv_s = f"{kv:.0f}%" if kv is not None else "--"
        cells.append(f"{kv_s:>7} /{pre:>5}")
    print(f"{run_label(key):<{w_run}}  {key[2]:<6}  " + "  ".join(cells))
print(LINE)

# Same output-token count per ratio is what makes the wall-clock comparable;
# report it once rather than per table.
print()
for key in keys:
    counts = {}
    for r in ratios:
        j = result(key, r)
        if j is None:
            continue
        counts[r] = j.get("benchmark", {}).get("total_output_tokens")
    if len(set(counts.values())) > 1:
        print(f" WARNING {run_label(key)}/{key[2]}: output tokens differ across "
              f"ratios {counts} -- the wall-clock columns are not comparable.")
PY
