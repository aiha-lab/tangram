# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KeyDiff scorer — key-similarity-based importance score (axis 2).

Scores post-RoPE keys by NEGATED cosine similarity to the average key
direction: a key close to the mean is redundant and goes first. Gate-free and
query-independent. Paper: https://arxiv.org/abs/2504.15364, ported from NVIDIA
KVpress (``keydiff_press.py``).

``forward`` anchors on the fresh chunk's own mean, ``score_cached`` on the whole
live cache. The ``anchor`` option picks the mean's spelling and BOTH use the
selected one: a scorer must not rank by one definition here and another there.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from vllm.v1.attention.compression.qk_scorer_base import (
    RESCORE_KEYS,
    CachedPositions,
    QKScorer,
)
from vllm.v1.attention.compression.scorer_options import ScorerOption


class KeyDiffScorer(QKScorer):
    """Reads ``key`` only; ``query`` is accepted to match the shared contract.
    Higher score = more distinctive, hence kept."""

    name = "keydiff"
    # A function of the cached keys alone, so every live position can be
    # rescored at every eviction -- what the paper's Eq. (8) asks for.
    rescores_cache = True
    rescore_inputs = (RESCORE_KEYS, )

    OPTIONS = (
        ScorerOption(
            "anchor", str, "unnormalized",
            "Which mean the keys are compared against: 'unnormalized' = mean of "
            "the raw keys mu(K), the paper's experimental setting; 'normalized' "
            "= mean of the L2-normalized directions mu(K-hat), Eq. (8) as "
            "written and what NVIDIA KVpress computes.",
            choices=("unnormalized", "normalized")),
    )

    def __init__(
        self,
        num_kv_heads: int,
        head_size: int,
        num_q_per_kv: int = 1,
        *,
        anchor: str = "unnormalized",
    ) -> None:
        # ``num_q_per_kv`` is part of the shared construction contract; KeyDiff
        # is query-independent and does not use it.
        del num_q_per_kv
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self._normalize_before_mean = anchor == "normalized"

    def _anchor(self, keys: torch.Tensor, dim: int) -> torch.Tensor:
        """Mean key along ``dim``, kept as a broadcastable axis.

        ``cosine_similarity`` normalizes both arguments, so only the anchor's
        DIRECTION reaches the score -- which is exactly what the two spellings
        disagree on: averaging raw keys lets a long key pull the mean towards
        itself, averaging directions gives every key the same pull.
        """
        if self._normalize_before_mean:
            keys = F.normalize(keys, p=2, dim=-1)
        return keys.mean(dim=dim, keepdim=True)

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
        # Key-only: the other arguments exist so every scorer shares one
        # call contract.
        del query, value, module, position_offset

        chunk_len = key.shape[0]
        # float32 for a stable mean/cosine; only precision differs.
        k = key.reshape(chunk_len, self.num_kv_heads, self.head_size).float()

        # Chunk anchor: the chunk is one KeyDiff block, matching KVpress
        # BlockPress(block_size=chunk), so the mean is over this chunk's keys.
        anchor = self._anchor(k, dim=0)                               # [1,H,d]

        # ``cosine_similarity`` re-normalizes both arguments, so the anchor's
        # magnitude never reaches the score. Negate so a distinctive key --
        # far from the mean direction -- scores high.
        score = -F.cosine_similarity(k, anchor, dim=-1)               # [T, H]
        return score.transpose(0, 1).contiguous()                    # [H, T]

    @torch.no_grad()
    def score_cached(self, cached: CachedPositions) -> torch.Tensor:
        """Score every live position of one head group from its cached keys.

        The paper's Eq. (8) as written: the anchor is the mean direction of ALL
        cached keys, so the ranking spans the whole cache rather than one chunk.
        That matters under a fixed budget, because ``forward`` gives each chunk
        its OWN anchor and scores measured against different references have no
        common scale. Recomputing here removes the problem and stores nothing,
        the keys being cached already.

        Returns ``[page_group_size, num_positions]`` float32, higher = keep.
        """
        # float32 over a cache-length reduction; the ranking is what matters.
        k = cached.require(RESCORE_KEYS).float()
        anchor = self._anchor(k, dim=1)
        return -F.cosine_similarity(k, anchor, dim=-1)
