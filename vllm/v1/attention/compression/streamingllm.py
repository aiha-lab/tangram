# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""StreamingLLM scorer — recency-based importance score (compression axis 2).

Produces the same ``[num_kv_heads, chunk_len]`` score contract every scorer
does, but from token *position* alone (it ignores query/key/value content).
The shared chunk machinery consumes the score identically.

Paper "Efficient Streaming Language Models with Attention Sinks"
(Xiao et al., https://arxiv.org/abs/2309.17453): keep the attention-sink
tokens plus the most recent tokens, evict the middle. tangram's shared
machinery already protects the sink (``n_sink_tokens``) and the recent window
(``window_size``) unconditionally, so this scorer only has to rank the
*eval region* (between sink and window) by recency — the most recent eval
tokens are kept first, extending the recent block. With the uniform selection
level this reproduces StreamingLLM exactly.

The score MUST be monotonic in the token's GLOBAL sequence position, not its
chunk-local position: the keep decision ranks the current chunk's body against
the carried-over previous-chunk window in one workspace (compressor.py
``prepare_keep_decision``), and those tokens span chunk boundaries. A
chunk-local ``arange`` would let an older previous-chunk token outrank a newer
current-chunk token (inverting recency), so the score is anchored at the
chunk's global start via ``position_offset``.

Because it is purely positional, StreamingLLM is best read as a recency
*baseline* — the reference line that content-aware scorers (SnapKV,
ExpectedAttention) must beat — rather than a content-aware method.
"""
from __future__ import annotations

import torch
from torch import nn


class StreamingLLMScorer(nn.Module):
    """One (stateless) instance shared across all compressible layers.

    Input:  ``query`` / ``key`` / ``value`` (all unused — accepted only to
            match the uniform qk scorer contract) and ``position_offset``, the
            global sequence position of the chunk's first token.
    Output: scores ``[num_kv_heads, chunk_len]`` (float32); higher = more
            recent (kept), lower = older (evicted first).
    """

    # Axis-2 dispatch: this scorer hooks the inner ``Attention`` (the qk hook),
    # so it shares the call signature even though it reads neither q nor k.
    consumes = "qk"
    name = "streamingllm"

    def __init__(self, num_kv_heads: int) -> None:
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
        # Global position of each token: a token later in the sequence scores
        # higher, so the uniform top-k keeps the most recent eval tokens. fp32
        # is exact for any real sequence position (< 2**24) and matches the
        # other scorers' fp32 output.
        positions = torch.arange(
            position_offset,
            position_offset + chunk_len,
            dtype=torch.float32,
            device=key.device,
        )
        # All heads share the same positional score (StreamingLLM is
        # head-agnostic); broadcast to the per-head contract. Materialize so the
        # returned tensor owns its storage (the score is stashed and later
        # concatenated / copied into the score workspace).
        return positions.unsqueeze(0).expand(
            self.num_kv_heads, chunk_len).contiguous()
