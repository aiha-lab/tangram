# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TOVA scorer — last-query attention importance score (axis 2).

Ported from NVIDIA KVpress (``tova_press.py``); paper "Transformers are
Multi-State RNNs" (https://arxiv.org/abs/2401.06104). Two properties define it
against SnapKV. LAST QUERY ONLY: a key's importance is how much the single most
recent query attends to it, where SnapKV averages a trailing window; "last" is
the last query of the current chunk, the chunk-local analogue of the
reference's last prompt token. HEAD-UNIFORM: the per-position attention is
averaged over ALL query heads into one score shared by every KV head, so every
head keeps the same positions, and with the uniform budget scope that
reproduces TOVA's single global KV policy.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from vllm.v1.attention.compression.qk_scorer_base import QKScorer


class TOVAScorer(QKScorer):
    """Scores are identical across heads: TOVA is head-uniform."""

    # Axis-2 dispatch: this scorer reads the inner ``Attention``'s q/k,
    # not the outer block's hidden_states.
    consumes = "qk"
    name = "tova"

    def __init__(
        self,
        num_kv_heads: int,
        head_size: int,
        num_q_per_kv: int = 1,
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
        # Last query's attention only; the rest is the shared contract.
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
