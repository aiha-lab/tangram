# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CLI: measure a per-(layer, head) KV retention profile from the live engine.

This is the first stage of the head-group clustering pipeline; the second stage
(``build_cluster_map.py``) consumes the ``.npz`` this emits. Together they let a
cluster map be built from nothing but a HuggingFace model id and (for the
Fast-KVzip scorer) a gate checkpoint -- both fetched from the Hub, no external
repository.

How it measures: the model is run through the REAL vLLM compression engine with
``page_group_size=1`` (so every head is its own group) and a retention observer
attached to each worker; the engine's own per-(layer, head) keep decision is
then read back out of the observer's dumps. Reading the engine's decision --
rather than replaying the model in HuggingFace transformers and reproducing the
keep decision in numpy -- makes the profile a single source of truth: it cannot
drift from what the engine actually does, and it works for every model the
engine supports, including eager-only models such as gpt-oss whose
query/key/value a transformers attention-capture hook cannot reach.

The emitted schema (``ratio_per_head`` etc.) matches what ``build_cluster_map``
reads.

Example:
    CUDA_VISIBLE_DEVICES=2 python -m tools.head_group_clustering.build_profile \
        --model Qwen/Qwen2.5-7B-Instruct-1M --scorer fastkvzip \
        --out tools/head_group_clustering/cluster_maps/fastkvzip/qwen2.5-7b-instruct-1m/profile.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoConfig, AutoTokenizer

# The pilot-data loader and sink template are the engine's own self-contained
# SCBench helpers (benchmarks/tangram/scbench_local.py). Add the tangram
# benchmarks dir to the path so it imports without packaging.
_BENCHMARKS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "benchmarks", "tangram")
if _BENCHMARKS_DIR not in sys.path:
    sys.path.insert(0, _BENCHMARKS_DIR)
from scbench_local import load_dataset_all, template  # noqa: E402

# Profiling fixes the first 32 prompt tokens as the protected attention sink and
# never evicts them (the head-group profiler's standing convention; independent
# of any per-model template length). The retention reported is context-only, so
# the exact sink count only has to be consistent between measurement and the
# context-only formula below -- it is not a tunable of the profile.
_PROFILE_SINK_TOKENS = 32


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="HuggingFace model id, e.g. Qwen/Qwen2.5-7B-Instruct-1M")
    p.add_argument("--gate-path", default="fastkvzip",
                   help="gate checkpoint: 'fastkvzip' (auto-resolve + Hub "
                        "download) or an explicit path. Default: fastkvzip. "
                        "Used only when --scorer fastkvzip.")
    p.add_argument("--scorer",
                   choices=["fastkvzip", "keydiff", "snapkv",
                            "expected_attention"],
                   default="fastkvzip",
                   help="which compression scorer's retention to profile: "
                        "'fastkvzip' (default, gate-scored hidden states), "
                        "'keydiff' (gate-free, post-RoPE key drift), 'snapkv' "
                        "(gate-free, observation-window attention; see "
                        "--snap-window/--snap-kernel), or 'expected_attention' "
                        "(gate-free, anticipated attention; see the --ea-* "
                        "options). The engine computes the chosen scorer's "
                        "per-head retention under the threshold set by --level, "
                        "so the profile matches the engine's selection for this "
                        "scorer. Name the output under the matching "
                        "<method>/<model> tree.")
    p.add_argument("--level", choices=["crosslayer_head", "perlayer_head"],
                   default="crosslayer_head",
                   help="threshold SCOPE whose per-head retention is profiled "
                        "(only the head-calibrated levels are profiled: the "
                        "profile is the per-head retention ranking that DRIVES "
                        "clustering, so it is measured before clusters exist; "
                        "the cluster-calibrated levels reuse these head-level "
                        "profiles via the scope-matched map). 'crosslayer_head' "
                        "(default): a single CROSS-layer global threshold "
                        "(strong layers keep more). 'perlayer_head': a SEPARATE "
                        "threshold per layer (AdaKV-style; every layer keeps its "
                        "own top-ratio fraction). The two produce different "
                        "per-head retention rankings, hence different cluster "
                        "maps -- name the output under the matching "
                        "<method>/<model> tree accordingly.")
    p.add_argument("--snap-window", type=int, default=32,
                   help="SnapKV observation window (trailing queries); matches "
                        "the engine's --compression-snap-window default. "
                        "Used only when --scorer snapkv.")
    p.add_argument("--snap-kernel", type=int, default=7,
                   help="SnapKV max-pool smoothing kernel; matches the engine's "
                        "--compression-snap-kernel default. Used only when "
                        "--scorer snapkv.")
    # ExpectedAttention hyperparameters (used only when --scorer
    # expected_attention); defaults mirror the engine / kvpress reference.
    p.add_argument("--ea-epsilon", type=float, default=1e-2,
                   help="ExpectedAttention value-norm floor (engine/kvpress "
                        "default 1e-2). Matches CacheConfig.compression_ea_epsilon.")
    p.add_argument("--ea-no-covariance", dest="ea_use_covariance",
                   action="store_false",
                   help="Disable the query-covariance term (default: on).")
    p.add_argument("--ea-no-vnorm", dest="ea_use_vnorm", action="store_false",
                   help="Disable value-norm reweighting (default: on).")
    p.add_argument("--ea-n-future-positions", type=int, default=512,
                   help="Future decode positions whose RoPE rotation is averaged "
                        "to anticipate future queries (engine default 512).")
    p.add_argument("--out", required=True, help="output profile .npz path")
    p.add_argument("--base-ratios", default="0.3,0.5,0.7",
                   help="comma-separated retention ratios (strictly increasing)")
    p.add_argument("--prefill-chunk", type=int, default=8192)
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   help="run the profiling engine at this TP degree. Match the "
                        "serving TP so the profile (and the cluster map built "
                        "from it) is per-rank: under TP each rank holds "
                        "num_kv_heads // TP heads, which is the head count the "
                        "runtime validates the map against. The per-(layer,head) "
                        "retention is averaged across ranks (one shared map "
                        "applies to every rank). Needed for qwen3-30b (TP=2).")
    p.add_argument("--window-size", type=int, default=4096)
    p.add_argument("--max-ctx-len", type=int, default=131072,
                   help="upper bound on per-sample context length (longer "
                        "samples are skipped). Keep large enough to include "
                        "representative long contexts -- the bundled "
                        "qwen2.5-7b profile used samples up to ~113k tokens.")
    p.add_argument("--dataset-mix", default=(
        "scbench_kv_short:20,scbench_repoqa:15,"
        "scbench_summary:5,scbench_many_shot:10"),
        help="<dataset>:<count> entries, comma-separated")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true",
                   help="parse inputs, skip GPU work (CLI smoke test)")
    return p.parse_args()


def parse_dataset_mix(spec: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, count = entry.split(":")
        out.append((name.strip(), int(count.strip())))
    return out


def parse_base_ratios(spec: str) -> list[float]:
    values = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not values:
        raise ValueError("--base-ratios must not be empty")
    if values != sorted(values) or len(values) != len(set(values)):
        raise ValueError("--base-ratios must be strictly increasing and unique")
    if any(v <= 0 or v > 1.0 for v in values):
        raise ValueError("--base-ratios must lie in (0, 1]")
    return values


# --------------------------------------------------------------------------- #
# Model structure (dense + SWA hybrid, generic)                                #
# --------------------------------------------------------------------------- #


def resolve_config(model_name: str):
    """Resolve (num_layers, num_kv_heads, num_static, static_layer_ids, is_swa)
    from the HF config alone -- no full model load.

    ``config.layer_types`` is the authoritative full-/sliding-attention signal
    (the same one the engine's static set is built from): hybrid models (gemma-3,
    gpt-oss) declare per-layer ``"full_attention"`` / ``"sliding_attention"``. A
    dense model has no ``layer_types``, so every layer is compressible and the
    static set is all layers. The engine scopes its keep decision to exactly the
    compressible layers, so this layer axis matches what is read back."""
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tcfg = getattr(cfg, "text_config", None) or cfg
    num_layers = int(tcfg.num_hidden_layers)
    num_kv_heads = int(tcfg.num_key_value_heads)
    layer_types = (getattr(tcfg, "layer_types", None)
                   or getattr(cfg, "layer_types", None))
    if layer_types is not None:
        static = [i for i, t in enumerate(layer_types)
                  if t == "full_attention"]
    else:
        static = list(range(num_layers))
    if not static:
        raise RuntimeError("no full-attention layers found in config")
    return num_layers, num_kv_heads, len(static), static, len(static) != num_layers


# --------------------------------------------------------------------------- #
# Sample selection                                                             #
# --------------------------------------------------------------------------- #


def load_datasets(mix: list[tuple[str, int]], tokenizer) -> dict:
    out: dict = {}
    for name, _ in mix:
        if name not in out:
            out[name] = load_dataset_all(name, tokenizer)
    return out


def select_samples(mix: list[tuple[str, int]], datasets: dict, tokenizer,
                   max_ctx_len: int, seed: int) -> list[tuple[str, int, int]]:
    """Deterministic (dataset_name, dataset_index, ctx_len) selection."""
    rng = random.Random(seed)
    out: list[tuple[str, int, int]] = []
    for name, want in mix:
        ds = datasets[name]
        order = list(range(len(ds)))
        rng.shuffle(order)
        chosen = 0
        for idx in order:
            ctx_len = len(tokenizer.encode(ds[idx]["context"],
                                           add_special_tokens=False))
            if ctx_len > max_ctx_len:
                continue
            out.append((name, idx, ctx_len))
            chosen += 1
            if chosen >= want:
                break
        if chosen < want:
            print(f"[warn] {name}: requested {want}, only {chosen} fit under "
                  f"max_ctx_len={max_ctx_len}")
    return out


# --------------------------------------------------------------------------- #
# Engine-driven measurement                                                    #
# --------------------------------------------------------------------------- #


def out_dump_dir(out_path, ratio):
    """Per-ratio scratch dir for engine retention dumps, beside the output."""
    return Path(str(out_path) + f".dump_r{ratio}")


def measure_all(args, tokenizer, mix, base_ratios):
    """Measure per-(static layer, head) retention from the REAL vLLM compression
    engine, one engine run per base ratio.

    Each run sets ``page_group_size=1`` (so each head is its own group) and
    ``compression_retention_dump=<dir>`` -- a typed config field that wires the
    engine's keep-decision observer to that directory inside every worker (see
    ``vllm/v1/attention/compression/profiling.py``). After generation the
    observer's per-request dumps are read back and converted to the same
    context-only retention the profile schema expects. All scorer
    hyperparameters are forwarded so non-default values (e.g. ``--ea-epsilon 0``)
    actually reach the engine; the engine ignores the ones irrelevant to the
    selected scorer."""
    from vllm import LLM, SamplingParams

    (num_layers, H, num_static, static_layer_ids, is_swa) = \
        resolve_config(args.model)
    tp = args.tensor_parallel_size
    if H % tp != 0:
        raise SystemExit(
            f"num_kv_heads ({H}) not divisible by --tensor-parallel-size ({tp})")
    # Under TP each rank holds H // tp KV heads; the profile (and the map built
    # from it) is per-rank, which is the head count the runtime validates against.
    h_per_rank = H // tp
    print(f"[load] L={num_layers} H_kv={H} (per_rank={h_per_rank} @ TP={tp}) "
          f"num_static={num_static} swa={is_swa} "
          f"static_layer_ids={static_layer_ids}")

    # Build the prompts: system prefix + each selected context. The sink template
    # ("qa") and SCBench pilot data are the engine's own self-contained helpers.
    prefix, _postfix = template(args.model, "qa")
    raw_datasets = load_datasets(mix, tokenizer)
    samples = select_samples(mix, raw_datasets, tokenizer,
                             args.max_ctx_len, args.seed)
    if not samples:
        raise SystemExit("no samples selected; check dataset names / max_ctx_len")
    prompts = [prefix + raw_datasets[ds][idx]["context"]
               for ds, idx, _pre in samples]
    print(f"[select] {len(samples)} samples: " + ", ".join(
        f"{name}={sum(1 for s in samples if s[0] == name)}" for name, _ in mix))

    max_model_len = args.max_ctx_len + 2048
    per_ratio_ratio, per_ratio_kept, per_ratio_meta = [], [], []
    for ratio in base_ratios:
        dump_dir = str(out_dump_dir(args.out, ratio))
        if os.path.isdir(dump_dir):
            shutil.rmtree(dump_dir)
        os.makedirs(dump_dir, exist_ok=True)
        print(f"[vllm] ratio={ratio} engine up (dump -> {dump_dir})", flush=True)
        llm = LLM(
            model=args.model, trust_remote_code=True, enforce_eager=True,
            tensor_parallel_size=tp,
            gpu_memory_utilization=0.85, max_model_len=max_model_len,
            enable_prefix_caching=False, max_num_seqs=8,
            page_group_size=1,
            compression_ratio=ratio, compression_scorer=args.scorer,
            compression_level=args.level,
            compression_window_size=args.window_size,
            compression_n_sink_tokens=_PROFILE_SINK_TOKENS,
            compression_floor_min=0,
            compression_chunk_size=args.prefill_chunk,
            compression_gate_path=args.gate_path,
            compression_snap_window=args.snap_window,
            compression_snap_kernel=args.snap_kernel,
            compression_ea_epsilon=args.ea_epsilon,
            compression_ea_use_covariance=args.ea_use_covariance,
            compression_ea_use_vnorm=args.ea_use_vnorm,
            compression_ea_n_future_positions=args.ea_n_future_positions,
            compression_retention_dump=dump_dir,
            # The engine requires max_num_batched_tokens == compression_chunk_size
            # when compression is on: it processes exactly one compression chunk
            # per prefill step. A context longer than the chunk is chunked
            # internally; a context that fits in one chunk is scored one-shot.
            max_num_batched_tokens=args.prefill_chunk)
        llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0))
        del llm
        # Read the engine's per-request keep decisions back, grouped by
        # (req, rank): under TP each rank dumps only its own KV-head shard, so a
        # request yields one file per rank. Dedup each (req, rank) by max
        # eval_len in case a request triggered more than one decision.
        by_key: dict[tuple, dict] = {}
        for f in glob.glob(os.path.join(dump_dir, "*.npz")):
            d = np.load(f, allow_pickle=False)
            key = (str(d["req"]), int(d["rank"]) if "rank" in d.files else 0)
            if key not in by_key or int(d["eval_len"]) > by_key[key]["eval_len"]:
                by_key[key] = {
                    "kept": d["kept"], "total": d["total"],
                    "sink": int(d["sink"]), "win": int(d["win"]),
                    "eval_len": int(d["eval_len"]),
                }
        if not by_key:
            raise SystemExit(
                f"no retention dumps for ratio={ratio}; every sample may have "
                "adjusted_ratio>=1 (too short) -- lower --window-size or raise "
                "--max-ctx-len.")
        ranks_of: dict[str, dict] = defaultdict(dict)
        for (req, rank), rec in by_key.items():
            ranks_of[req][rank] = rec
        ratios, kepts, metas = [], [], []
        for req, ranks in ranks_of.items():
            # Per-rank context-only retention, then average across ranks. The
            # runtime applies ONE shared [num_static, h_per_rank] map to every
            # rank, so the profile head axis is per-rank (rank r's local head h
            # is physical head r*h_per_rank + h); averaging gives the shared map
            # a representative ranking. For TP=1 this is a no-op (one rank).
            per_rank_r, per_rank_kept = [], []
            for rank in sorted(ranks):
                rec = ranks[rank]
                kept = rec["kept"].astype(np.int64)    # [num_static, h_per_rank]
                total = rec["total"].astype(np.int64)
                sink, win = rec["sink"], rec["win"]
                # (kept - sink - window) / (total seen - sink): the fraction of
                # the evictable context each head keeps, independent of the
                # always-kept sink and recent window.
                ctx = np.maximum(total - sink, 1)
                r = np.clip(kept - sink - win, 0, None).astype(np.float32) / ctx
                per_rank_r.append(r.clip(0.0, 1.0))
                per_rank_kept.append(kept)
            ratios.append(np.mean(np.stack(per_rank_r, 0), axis=0))   # [L, h_pr]
            kepts.append(np.mean(np.stack(per_rank_kept, 0), axis=0).astype(
                np.int32))
            rec0 = ranks[min(ranks)]
            total0, sink0 = rec0["total"].astype(np.int64), rec0["sink"]
            metas.append([int(total0[0, 0] - sink0), sink0,
                          int(total0[0, 0]), rec0["win"]])
        got_h = ratios[0].shape[-1]
        if got_h != h_per_rank:
            raise SystemExit(
                f"dump head dim {got_h} != expected per-rank {h_per_rank} "
                f"(H={H}, TP={tp}); check --tensor-parallel-size.")
        per_ratio_ratio.append(np.stack(ratios, axis=0))   # [S, L, h_per_rank]
        per_ratio_kept.append(np.stack(kepts, axis=0))
        per_ratio_meta.append(np.asarray(metas, dtype=np.int32))
        print(f"[vllm] ratio={ratio} collected {len(ranks_of)} samples "
              f"({len(by_key)} rank-dumps)", flush=True)

    # Equalise the per-ratio sample counts (some samples may degenerate to
    # adjusted_ratio>=1 at higher ratios and not dump) so the [R, S, L, H]
    # stack is rectangular; truncate to the common minimum.
    s_min = min(a.shape[0] for a in per_ratio_ratio)
    raw_ratio = np.stack([a[:s_min] for a in per_ratio_ratio], axis=0)
    raw_kept = np.stack([a[:s_min] for a in per_ratio_kept], axis=0)
    sample_meta = per_ratio_meta[0][:s_min]
    return (raw_ratio, raw_kept, sample_meta, samples, num_layers, h_per_rank,
            num_static, static_layer_ids, is_swa)


# --------------------------------------------------------------------------- #
# Aggregation + serialization                                                  #
# --------------------------------------------------------------------------- #


def aggregate_and_write(out_path, raw_ratio, raw_kept, sample_meta, samples,
                        base_ratios, args, num_layers, num_kv_heads, num_static,
                        static_layer_ids, is_swa) -> int:
    """Aggregate per-sample retention into the ranking profile and write the
    profile ``.npz``. ``build_cluster_map`` and the engine read the result; the
    layer axis is the compressible (full-attention) layer space (``num_static``,
    == ``num_layers`` for a dense model)."""
    n_samples = raw_ratio.shape[1]
    # Aggregate over the sample axis (median ranking is what clustering uses).
    ratio_per_head = np.median(raw_ratio, axis=1).astype(np.float32)
    ratio_p90_per_head = np.percentile(raw_ratio, 90, axis=1).astype(np.float32)
    ratio_std_per_head = np.std(raw_ratio, axis=1).astype(np.float32)
    coverage = float((ratio_std_per_head.reshape(-1) < 0.05).mean())
    print(f"[aggregate] model-intrinsic coverage={coverage:.2%}")

    meta = {
        "model_name": args.model,
        # All profiles are now measured by reading the live engine's keep
        # decision (single source of truth); recorded for provenance.
        "backend": "vllm",
        "num_hidden_layers": num_layers,
        "num_static_layers": num_static,
        "static_layer_ids": list(static_layer_ids),
        "is_swa": is_swa,
        # Per-rank under TP (== model total when tensor_parallel_size == 1);
        # this is the head count the runtime validates the cluster map against.
        "num_kv_heads": num_kv_heads,
        "tensor_parallel_size": args.tensor_parallel_size,
        "sample_count": n_samples,
        "sample_sources": sorted({s[0] for s in samples}),
        "ctx_len_min": int(sample_meta[:, 0].min()),
        "ctx_len_max": int(sample_meta[:, 0].max()),
        "base_ratios": list(base_ratios),
        "window_size": args.window_size,
        "prefill_chunk": args.prefill_chunk,
        "scorer": args.scorer,
        "level": args.level,
        "snap_window": args.snap_window if args.scorer == "snapkv" else None,
        "snap_kernel": args.snap_kernel if args.scorer == "snapkv" else None,
        "fastkvzip_gate_name": args.gate_path if args.scorer == "fastkvzip"
        else None,
        "tool": "tools/head_group_clustering/build_profile.py",
        "tool_version": "v1",
        "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_intrinsic_coverage": coverage,
        "seed": args.seed,
        "dataset_mix": args.dataset_mix,
        "sink_template_task": "qa",
        "transformers_version": __import__("transformers").__version__,
    }
    meta_bytes = json.dumps(meta, indent=2).encode("utf-8")
    np.savez(
        out_path,
        ratio_per_head=ratio_per_head,
        ratio_p90_per_head=ratio_p90_per_head,
        ratio_std_per_head=ratio_std_per_head,
        base_ratios=np.asarray(base_ratios, dtype=np.float32),
        # Top-level so consumers (e.g. fig6.py) map the compressed layer axis
        # back to physical layers without decoding meta; for a dense model this
        # is just range(num_layers).
        static_layer_ids=np.asarray(static_layer_ids, dtype=np.int64),
        meta=np.frombuffer(meta_bytes, dtype=np.uint8),
        raw_kept=raw_kept,
        raw_ratio=raw_ratio,
        sample_meta=sample_meta,
    )
    print(f"[done] wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return 0


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> int:
    args = parse_args()
    base_ratios = parse_base_ratios(args.base_ratios)
    mix = parse_dataset_mix(args.dataset_mix)

    np.random.seed(args.seed)
    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"[dry-run] model={args.model} scorer={args.scorer} "
              f"level={args.level} gate={args.gate_path} "
              f"base_ratios={base_ratios} mix={mix} out={out_path}")
        return 0

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    (raw_ratio, raw_kept, sample_meta, samples, L, H, num_static,
     static_layer_ids, is_swa) = measure_all(args, tokenizer, mix, base_ratios)
    return aggregate_and_write(
        out_path, raw_ratio, raw_kept, sample_meta, samples, base_ratios,
        args, L, H, num_static, static_layer_ids, is_swa)


if __name__ == "__main__":
    raise SystemExit(main())
