# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retention profiling for head-group cluster-map construction.

The head-group clustering tool (``tools/head_group_clustering``) needs the
per-(layer, head) fraction of KV cache each head retains under non-uniform
compression. That fraction is precisely what the engine's keep decision already
computes, so this module lets an offline profiler read those decisions back out
of a live engine instead of re-deriving them.

Reading from the live engine is the only approach that works for every model.
The alternative — replaying the model in HuggingFace transformers and capturing
query/key/value through the attention-function registry — cannot reach the
query/key/value of eager-only models (for example gpt-oss, whose attention sinks
have no SDPA kernel, so transformers never dispatches its attention through the
registry the capture hooks rely on). The engine, by contrast, scores every model
identically through its own attention path.

Wiring is config-driven (no engine entanglement beyond one optional hook): set
``CacheConfig.compression_retention_dump`` to a directory and the worker attaches
the observer to its compressor at construction (see ``gpu_model_runner``). From
the ``LLM`` entrypoint that is just a keyword argument:

    from vllm import LLM

    llm = LLM(model=..., enable_compression=True, page_group_size=1,
              compression_retention_dump=dump_dir, ...)
    llm.generate(prompts, ...)          # each keep decision is written to disk
    # then aggregate dump_dir into a profile (build_profile.py --backend vllm)

``page_group_size=1`` makes every ``(layer, group)`` a single head, so the dumped
``kept`` / ``total`` arrays are per-(layer, head).
"""
from __future__ import annotations

import os

import numpy as np


class RetentionProfileObserver:
    """Persist each per-request keep decision to ``<dump_dir>/<req_id>_<seq>.npz``.

    One file per recorded decision keeps the writer crash-safe and lets the
    aggregator simply glob the directory. The aggregator recovers context-only
    retention per head as ``(kept - sink - window) / (total - sink)`` — the same
    quantity the transformers profiling path emits — so profiles built either
    way are interchangeable.

    The writer runs inside the engine worker process; the aggregator reads the
    files from the driver process. Both share the local filesystem, so this is
    single-node only (the head-group profiler always runs single-node).

    Under tensor parallelism every rank runs its own observer and they share the
    dump directory, so the filename carries the ``rank``: a rank sees only its
    own ``num_kv_heads // tensor_parallel_size`` heads, and without the rank tag
    two ranks would write the same ``<req>_<seq>`` name and clobber each other.
    The aggregator groups files by ``(req, rank)`` to reassemble per-rank heads.
    """

    def __init__(self, dump_dir: str, rank: int = 0) -> None:
        self._dump_dir = dump_dir
        self._rank = rank
        # Monotonic per-observer counter disambiguating multiple decisions for
        # one request id (e.g. were a request ever compressed more than once).
        self._seq = 0
        os.makedirs(dump_dir, exist_ok=True)

    def record(
        self,
        req_id: str,
        *,
        kept_lengths: np.ndarray,
        total_seen: np.ndarray,
        sink_size: int,
        win_size: int,
        eval_len: int,
    ) -> None:
        path = os.path.join(
            self._dump_dir, f"{req_id}_r{self._rank}_{self._seq}.npz")
        self._seq += 1
        np.savez(
            path,
            kept=kept_lengths.astype(np.int64),
            total=total_seen.astype(np.int64),
            sink=np.int64(sink_size),
            win=np.int64(win_size),
            eval_len=np.int64(eval_len),
            req=np.array(str(req_id)),
            rank=np.int64(self._rank),
        )
