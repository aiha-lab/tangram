# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SnapKV scorer — chunk-local attention-based importance score (axis 2).

Produces the same ``[num_kv_heads, chunk_len]`` score contract the gate does,
but from the model's post-RoPE query/key of the current chunk instead of
hidden_states. The shared chunk machinery (sink / window / lock-in /
adjusted_ratio / executor) consumes the score identically.

Ported from the reference ``baseline.py:SnapKV`` (KVzip). Because tangram is
chunk-based, the ``key`` seen here is only the current chunk's keys, so the
reference's sink-prepend collapses (``sink == 0``): the scorer emits raw
chunk-position scores and the shared machinery protects sink/window.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from vllm.v1.attention.compression.qk_scorer_base import QKScorer


class SnapKVScorer(QKScorer):
    """One (stateless) instance shared across all compressible layers.

    Input:  ``query [T, num_kv_heads * num_q_per_kv * head_size]`` and
            ``key   [T, num_kv_heads * head_size]`` (post-RoPE, token-major
            flatten) for one request's chunk, or the equivalent 3-D views.
    Output: scores ``[num_kv_heads, T]`` (float32), higher = more important.
    """

    # Axis-2 dispatch: this scorer reads the inner ``Attention``'s q/k,
    # not the outer block's hidden_states.
    consumes = "qk"
    name = "snapkv"

    def __init__(
        self,
        num_kv_heads: int,
        num_q_per_kv: int,
        head_size: int,
        snap_window: int,
        snap_kernel: int,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.num_q_per_kv = num_q_per_kv
        self.head_size = head_size
        self.snap_window = snap_window
        self.snap_kernel = snap_kernel
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
        # SnapKV scores from the observation-window attention over query/key
        # only; ``value`` / ``module`` / ``position_offset`` are part of the
        # shared qk scorer contract but unused here.
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
