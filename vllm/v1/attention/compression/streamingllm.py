# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""StreamingLLM scorer — recency-based importance score (axis 2).

Scores by token position alone, ignoring query/key/value content. Paper:
"Efficient Streaming Language Models with Attention Sinks"
(https://arxiv.org/abs/2309.17453) -- keep the sink plus the most recent tokens,
evict the middle. Both of those are already protected unconditionally, so this
scorer only ranks the eval region between them by recency, extending the recent
block; with the uniform budget scope that reproduces StreamingLLM exactly.

The score MUST be monotonic in the token's GLOBAL sequence position, hence the
anchor at ``position_offset``: the keep decision ranks the fresh chunk against
the carried previous window in one workspace, so a chunk-local ``arange`` would
let an older carried token outrank a newer one and invert recency.
"""
from __future__ import annotations

import torch
from torch import nn
from vllm.v1.attention.compression.qk_scorer_base import QKScorer


class StreamingLLMScorer(QKScorer):
    """Reads only ``position_offset``, the global sequence position of the
    chunk's first token; query/key/value are accepted to match the shared
    contract. Higher score = more recent, hence kept."""

    # Axis-2 dispatch: this scorer uses the query/key delivery path, so it
    # shares the call signature even though it reads neither q nor k.
    consumes = "qk"
    name = "streamingllm"

    def __init__(
        self,
        num_kv_heads: int,
        head_size: int = 0,
        num_q_per_kv: int = 1,
    ) -> None:
        # Part of the shared construction contract; a positional score needs
        # neither.
        del head_size, num_q_per_kv
        super().__init__()
        self.num_kv_heads = num_kv_heads

    @torch.no_grad()
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor | None = None,
        *,
        module: nn.Module | None = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        # Position-only score; all tensor inputs and ``module`` are unused.
        # ``key`` supplies just the chunk length and the target device.
        del query, value, module

        chunk_len = key.shape[0]
        # Later in the sequence scores higher, so the top-k keeps the most
        # recent eval tokens. fp32 is exact below 2**24 positions and matches
        # every other scorer's output dtype.
        positions = torch.arange(
            position_offset,
            position_offset + chunk_len,
            dtype=torch.float32,
            device=key.device,
        )
        # Head-agnostic, so broadcast one score to the per-head contract.
        # Materialize: the result is stashed and later copied, so it must own
        # its storage.
        return positions.unsqueeze(0).expand(
            self.num_kv_heads, chunk_len).contiguous()
