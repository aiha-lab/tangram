# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retention profiling for head-group cluster-map construction.

``tools/head_group_clustering`` needs the per-(layer, head) fraction of KV each
head retains under non-uniform compression, which is exactly what the engine's
keep decision already computes. This observer writes those decisions out so an
offline profiler can read them back instead of re-deriving them. Replaying the
model in HuggingFace transformers cannot: eager-only models (gpt-oss, whose
attention sinks have no SDPA kernel) never dispatch through the registry a
capture hook would need.

Wiring is one config field. Set ``CacheConfig.compression_retention_dump`` to a
directory -- ``LLM(..., compression_retention_dump=dump_dir)`` -- and the worker
attaches the observer at construction. Use ``page_group_size=1`` so every
(layer, group) is a single head and the dumped ``kept`` / ``total`` arrays are
per-(layer, head).
"""
from __future__ import annotations

import os

import numpy as np


class RetentionProfileObserver:
    """Persist each per-request keep decision to ``<dump_dir>/<req_id>_<seq>.npz``.

    One file per decision keeps the writer crash-safe and lets the aggregator
    glob the directory. It recovers context-only retention per head as
    ``(kept - sink - window) / (total - sink)``, the same quantity the
    transformers path emits, so profiles built either way are interchangeable.
    Writer in the worker, aggregator in the driver, both on the local
    filesystem: single-node only.

    The filename carries the ``rank`` because each rank observes only its own
    head shard into a shared directory; the aggregator groups by
    ``(req, rank)``.
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
