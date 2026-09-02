# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SnapKV scorer — chunk-local attention-based importance score (axis 2).

Ported from the KVzip reference ``baseline.py:SnapKV``. Because tangram is
chunk-based the ``key`` seen here is only the current chunk's, so the
reference's sink-prepend collapses to ``sink == 0``: this scorer emits raw
chunk-position scores and the shared machinery protects the sink and window.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from vllm.v1.attention.compression.qk_scorer_base import QKScorer
from vllm.v1.attention.compression.scorer_options import ScorerOption


class SnapKVScorer(QKScorer):
    """Accepts the token-major flattened query/key or the equivalent 3-D
    views."""

    name = "snapkv"

    OPTIONS = (
        ScorerOption(
            "window", int, 32,
            "Trailing queries used as the observation window when scoring a "
            "chunk. Distinct from compression_window_size (the always-kept "
            "recent region); auto-shrinks to 16 for chunks shorter than 1000, "
            "matching the reference.",
            requirement=("a positive integer", lambda v: v > 0)),
        ScorerOption(
            "kernel", int, 7,
            "Odd max-pool1d kernel size smoothing the observation-window "
            "attention before ranking.",
            requirement=("a positive odd integer",
                         lambda v: v > 0 and v % 2 == 1)),
    )

    def __init__(
        self,
        num_kv_heads: int,
        head_size: int,
        num_q_per_kv: int = 1,
        *,
        window: int = 32,
        kernel: int = 7,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.num_q_per_kv = num_q_per_kv
        self.head_size = head_size
        self.snap_window = window
        self.snap_kernel = kernel
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
        # Observation-window attention over query/key only; the rest is the
        # shared contract.
        del value, module, position_offset
        num_kv_heads = self.num_kv_heads
        num_q_per_kv = self.num_q_per_kv
        head_size = self.head_size

        chunk_len = query.shape[0]
        # [T, num_kv_heads, num_q_per_kv, head_size] / [T, num_kv_heads, d].
        q = query.reshape(chunk_len, num_kv_heads, num_q_per_kv, head_size)
        k = key.reshape(chunk_len, num_kv_heads, head_size)

        # Observation window: trailing queries only. Short chunks shrink the
        # window to 16, matching the reference's adaptive behaviour.
        window = self.snap_window if chunk_len >= 1000 else min(16, chunk_len)

        # [window, num_kv_heads, num_q_per_kv, d] -> [num_kv_heads, group, w, d]
        q = q[chunk_len - window:].permute(1, 2, 0, 3)
        # [num_kv_heads, d, T]
        k_t = k.permute(1, 2, 0)

        # [num_kv_heads, group, w, T]; GQA group reduced by amax (reference).
        attn = torch.matmul(q, k_t.unsqueeze(1)) / self._scale
        attn = attn.amax(dim=1)                              # [num_kv_heads, w, T]

        # softmax over key positions, averaged over the query window.
        weights = torch.softmax(
            attn, dim=-1, dtype=torch.float32).mean(dim=-2)  # [num_kv_heads, T]

        # Smooth so a sharply-attended token also protects its neighbours.
        score = F.max_pool1d(
            weights,
            kernel_size=self.snap_kernel,
            padding=self.snap_kernel // 2,
            stride=1,
        )
        return score                                          # [num_kv_heads, T]
