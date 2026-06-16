# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TOVA scorer — last-query attention importance score (compression axis 2).

Produces the same ``[num_kv_heads, chunk_len]`` score contract every scorer
does, from the model's post-RoPE query/key of the current chunk. The shared
chunk machinery consumes the score identically.

Ported from NVIDIA KVpress (``kvpress/presses/tova_press.py``); paper
"Transformers are Multi-State RNNs" (Oren et al., https://arxiv.org/abs/2401.06104).
The reference docstring: "Uses attention weights of the last token (averaged
across heads) to estimate importance of previous key-value pairs."

Two properties define TOVA and separate it from SnapKV:
* **Last query only** — the importance of a key is how much the single most
  recent query attends to it (SnapKV averages a trailing observation window).
* **Head-uniform** — the per-position attention is averaged across ALL query
  heads into one score, then shared by every KV head, so every head keeps the
  same positions (SnapKV keeps per-head positions). With the uniform selection
  level this reproduces TOVA's single global KV policy.

Like tangram's SnapKV scorer, the attention is computed from the chunk's
post-RoPE query/key supplied by the qk hook (the reference recomputes the
query from hidden_states; tangram already has the post-RoPE query, so it skips
that recomputation — same attention). "Last query" is the last query of the
current chunk, the chunk-local analogue of the reference's last prompt token.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class TOVAScorer(nn.Module):
    """One (stateless) instance shared across all compressible layers.

    Input:  ``query [T, num_kv_heads * num_q_per_kv * head_size]`` and
            ``key   [T, num_kv_heads * head_size]`` (post-RoPE, token-major
            flatten) for one request's chunk.
    Output: scores ``[num_kv_heads, T]`` (float32), higher = more important;
            identical across heads (TOVA is head-uniform).
    """

    # Axis-2 dispatch: this scorer hooks the inner ``Attention`` (sees q/k),
    # not the outer block (which sees hidden_states).
    consumes = "qk"
    name = "tova"

    def __init__(
        self,
        num_kv_heads: int,
        num_q_per_kv: int,
        head_size: int,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.num_q_per_kv = num_q_per_kv
        self.head_size = head_size
        self._scale = math.sqrt(head_size)

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
        # TOVA scores from the last query's attention only; ``value`` /
        # ``module`` / ``position_offset`` are part of the shared qk contract
        # but unused here.
        del value, module, position_offset

        num_kv_heads = self.num_kv_heads
        num_q_per_kv = self.num_q_per_kv
        head_size = self.head_size

        chunk_len = query.shape[0]
        # [T, num_kv_heads, num_q_per_kv, head_size] / [T, num_kv_heads, d].
        q = query.reshape(chunk_len, num_kv_heads, num_q_per_kv, head_size)
        k = key.reshape(chunk_len, num_kv_heads, head_size)

        # Last query only (window == 1). The last query attends causally to all
        # keys in the chunk, so no masking is needed.
        # [num_kv_heads, num_q_per_kv, head_size]
        q_last = q[chunk_len - 1]
        # [num_kv_heads, head_size, T]
        k_t = k.permute(1, 2, 0)

        # [num_kv_heads, num_q_per_kv, T]: each query head's attention logits.
        attn = torch.matmul(q_last, k_t) / self._scale
        weights = torch.softmax(attn, dim=-1, dtype=torch.float32)

        # Average over ALL query heads (kv heads x group) → one score per
        # position, then share it across every KV head (TOVA is head-uniform).
        score = weights.mean(dim=(0, 1))                     # [T]
        return score.unsqueeze(0).expand(
            num_kv_heads, chunk_len).contiguous()            # [num_kv_heads, T]
