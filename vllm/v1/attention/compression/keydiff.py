# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KeyDiff scorer — key-similarity-based importance score (compression axis 2).

Produces the same ``[num_kv_heads, chunk_len]`` score contract every scorer
does, from the model's post-RoPE keys of the current chunk. The shared chunk
machinery (sink / window / lock-in / adjusted_ratio / selection level /
executor) consumes the score identically.

Ported from NVIDIA KVpress (``kvpress/presses/keydiff_press.py``); paper
"KeyDiff: Key Similarity-Based KV Cache Eviction" (https://arxiv.org/abs/2504.15364).
Reference: ``fastkvzip-accuracy-reproduce/prefill/attention/baseline.py:KeyDiff``.

Intuition: within a chunk, the keys whose direction is closest to the chunk's
average key direction are the least distinctive (most redundant), so they carry
the least information and are evicted first. The score is the NEGATED cosine
similarity to that average direction — higher (less similar to the mean) means
more distinctive, hence kept. KeyDiff is gate-free and query-independent (it
only reads keys); like the reference it defaults to a uniform per-head budget
(pair-head), but the selection level stays an orthogonal knob.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class KeyDiffScorer(nn.Module):
    """One (stateless) instance shared across all compressible layers.

    Input:  ``query [T, num_kv_heads * num_q_per_kv * head_size]`` (unused —
            present only to match the shared query/key scorer call signature)
            and ``key [T, num_kv_heads * head_size]`` (post-RoPE, token-major
            flatten) for one request's chunk.
    Output: scores ``[num_kv_heads, T]`` (float32), higher = more distinctive
            (kept); lower = more redundant (evicted first).
    """

    # Axis-2 dispatch: this scorer hooks the inner ``Attention`` (sees q/k),
    # not the outer block (which sees hidden_states).
    consumes = "qk"
    name = "keydiff"

    def __init__(
        self,
        num_kv_heads: int,
        head_size: int,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size

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
        # KeyDiff is key-only; ``query`` / ``value`` / ``module`` /
        # ``position_offset`` are accepted only so the qk scorer hook can call
        # every scorer with the same uniform contract.
        del query, value, module, position_offset

        chunk_len = key.shape[0]
        # [T, num_kv_heads, head_size]. float32 for a stable mean / cosine
        # (ranking-faithful to the reference; only the precision differs).
        k = key.reshape(chunk_len, self.num_kv_heads, self.head_size).float()

        # Chunk anchor = mean of the L2-normalized key directions (the chunk is
        # one KeyDiff block, matching KVpress BlockPress(block_size=chunk)).
        anchor = F.normalize(k, p=2, dim=-1).mean(dim=0, keepdim=True)  # [1,H,d]

        # ``cosine_similarity`` re-normalizes ``k`` internally, so passing raw
        # keys reproduces the reference exactly (anchor stays unnormalized).
        # Negate so distinctive keys (far from the mean direction) score high.
        score = -F.cosine_similarity(k, anchor, dim=-1)               # [T, H]
        return score.transpose(0, 1).contiguous()                    # [H, T]
