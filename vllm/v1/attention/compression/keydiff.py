# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KeyDiff scorer — key-similarity-based importance score (compression axis 2).

Produces the same ``[num_kv_heads, chunk_len]`` score contract every scorer
does, from the model's post-RoPE keys of the current chunk. The shared chunk
machinery (sink / window / lock-in / adjusted_ratio / budget scope /
executor) consumes the score identically.

Ported from NVIDIA KVpress (``kvpress/presses/keydiff_press.py``); paper
"KeyDiff: Key Similarity-Based KV Cache Eviction" (https://arxiv.org/abs/2504.15364).
Reference: ``fastkvzip-accuracy-reproduce/prefill/attention/baseline.py:KeyDiff``.

Intuition: the keys whose direction is closest to the average key direction are
the least distinctive (most redundant), so they carry the least information and
are evicted first. The score is the NEGATED cosine similarity to that average
direction — higher (less similar to the mean) means more distinctive, hence
kept. KeyDiff is gate-free and query-independent (it only reads keys); like the
reference it defaults to a uniform per-head budget (pair-head), but the
budget scope stays an orthogonal knob.

Two entry points, differing only in WHICH keys the average is taken over:
``forward`` scores the fresh chunk against the chunk's own mean (the
chunk-as-one-block variant), and ``score_cached`` scores every live cache
position against the mean of the whole cache, which is the paper's Eq. (8) and
what a fixed KV budget needs.

The anchor itself has two published spellings and the ``anchor`` option selects
between them (both entry points use the selected one — a scorer must not rank
by one definition here and another there):

* ``"unnormalized"`` (default) — the mean of the raw keys, mu(K). The paper's
  section 3.2 states "We evaluate the efficient KeyDiff described in Figure 3
  using unnormalized keys k in all subsequent sections", so this is what its
  reported numbers use.
* ``"normalized"`` — the mean of the L2-normalized key directions, mu(K-hat),
  which is how Eq. (8) is written. The paper reports the two as equally
  accurate (Table 15); NVIDIA KVpress computes this one.
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
    """One (stateless) instance shared across all compressible layers.

    Input:  ``query [T, num_kv_heads * num_q_per_kv * head_size]`` (unused —
            present only to match the shared query/key scorer call signature)
            and ``key [T, num_kv_heads * head_size]`` (post-RoPE, token-major
            flatten) for one request's chunk.
    Output: scores ``[num_kv_heads, T]`` (float32), higher = more distinctive
            (kept); lower = more redundant (evicted first).
    """

    # Axis-2 dispatch: this scorer reads the inner ``Attention``'s q/k,
    # not the outer block's hidden_states.
    consumes = "qk"
    name = "keydiff"
    # The score is a function of the cached keys alone, so it can be recomputed
    # for every live position at every eviction — which is what the paper's
    # Eq. (8) actually asks for (see ``score_cached``). Keys are all it needs.
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
        """Mean key of ``keys`` along ``dim``, kept as a broadcastable axis.

        ``cosine_similarity`` normalizes both of its arguments, so the anchor's
        magnitude never reaches the score — only its DIRECTION does, and that is
        what the two spellings disagree on: averaging raw keys lets a long key
        pull the mean towards itself, averaging directions gives every key the
        same pull.
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
        # KeyDiff is key-only; ``query`` / ``value`` / ``module`` /
        # ``position_offset`` are accepted only so the query/key scorer path
        # can call every scorer with the same uniform contract.
        del query, value, module, position_offset

        chunk_len = key.shape[0]
        # [T, num_kv_heads, head_size]. float32 for a stable mean / cosine
        # (ranking-faithful to the reference; only the precision differs).
        k = key.reshape(chunk_len, self.num_kv_heads, self.head_size).float()

        # Chunk anchor: the chunk is one KeyDiff block, matching KVpress
        # BlockPress(block_size=chunk), so the mean is over this chunk's keys.
        anchor = self._anchor(k, dim=0)                               # [1,H,d]

        # ``cosine_similarity`` re-normalizes both arguments, so the
        # anchor's magnitude never reaches the score — the two anchor
        # formulas differ only in direction.
        # Negate so distinctive keys (far from the mean direction) score high.
        score = -F.cosine_similarity(k, anchor, dim=-1)               # [T, H]
        return score.transpose(0, 1).contiguous()                    # [H, T]

    @torch.no_grad()
    def score_cached(self, cached: CachedPositions) -> torch.Tensor:
        """Score every live position of one head group from its cached keys.

        This is the paper's Eq. (8) as written: the anchor is the mean direction
        of ALL keys currently in the cache and every cached key is scored against
        it, so the ranking is over the whole cache rather than within one chunk.

        The distinction matters under a fixed KV budget. ``forward`` above takes
        the chunk as one KeyDiff block (matching KVpress
        ``BlockPress(block_size=chunk)``), which gives each chunk its OWN anchor;
        scores from different chunks are then measured against different
        references and ranking them together has no common scale. Recomputing
        here removes that problem entirely — and needs no stored statistics,
        because the keys the score depends on are already in the cache.

        Args:
            cached: the group's live slots; KeyDiff declares ``keys`` only.

        Returns:
            ``[page_group_size, num_positions]`` float32, higher = keep.
        """
        # float32 for a stable mean / cosine over a cache-length reduction; the
        # ranking, not the magnitude, is what the keep decision consumes.
        k = cached.require(RESCORE_KEYS).float()
        anchor = self._anchor(k, dim=1)
        return -F.cosine_similarity(k, anchor, dim=-1)
