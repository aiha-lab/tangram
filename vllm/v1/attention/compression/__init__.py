# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Non-uniform KV cache compression subsystem.

Vocabulary for the whole package. No module below redefines these.

Three axes configure a run: the *budget scope* (the range a keep count is
balanced over -- ``uniform`` / ``layer`` default / ``global``), the *scorer*,
and the *regime* (``ratio``, a fraction of the prompt with lock-in, or
``budget``, an absolute count without).

A *member* is one (layer, KV head), row ``layer * per-rank KV heads + head``. A
*cluster* (== head group) is the members sharing one page table, and is the
unit of both compression and block ownership. A member's *column* is its slot
in the cluster page. A *virtual block* is
``physical_block * page_group_size + column``, what the attention kernel and
``reshape_and_cache`` receive. A *cache slot* is a position in a cluster's
compacted KV, the index the block table and the executor address.

The invariant that shapes most of this code: members of one cluster share ONE
length and differ only in WHICH positions they keep.

A cluster's cache in slot order is *sink* (always-kept head), *locked* (fixed
by an earlier chunk, ratio regime only), *eval region* (what competes on score
this chunk) and *tail* (always-kept recent positions); ``kept_length`` is their
block-aligned total.

A *cluster map* is an offline ``.npz`` of ``cluster_of`` / ``column_of``, and
``None`` means auto-resolve, NOT identity. *Identity grouping* is the degraded
fallback ``h -> cluster h // page_group_size``, column ``h % page_group_size``.

A *static* layer index counts compressible (full-attention) layers only, a
*physical* one is the model's real index; they differ only for hybrids.
"""
from vllm.v1.attention.compression.compressor import KVCompressor
from vllm.v1.attention.compression.executor import (
    CompressionExecutor,
    CompressionMetadata,
)
from vllm.v1.attention.compression.eviction_regime import ChunkParams

__all__ = [
    "KVCompressor",
    "CompressionExecutor",
    "CompressionMetadata",
    "ChunkParams",
]
