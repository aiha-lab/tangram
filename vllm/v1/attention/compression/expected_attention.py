# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ExpectedAttention scorer — analytic expected-attention score (axis 2).

Produces the same ``[num_kv_heads, chunk_len]`` score contract every scorer
does. Ported faithfully from NVIDIA KVpress
(``kvpress/presses/expected_attention_press.py``): it estimates, WITHOUT
materialising an attention matrix, the attention each key is expected to
receive from FUTURE (decode) queries, then (optionally) reweights by the value
norm.

Reference algorithm (per QUERY head, then averaged over the GQA group):
  1. From the PRE-RoPE queries: mean ``mu`` and covariance ``Sigma`` over the
     observed query positions (the first few chunk queries are dropped as
     outliers; covariance is normalised by the number of query positions).
  2. Apply the average RoPE rotation ``R`` of the next ``n_future_positions``
     positions to anticipate where future queries sit.
  3. logit(key) = (R·mu)·k / sqrt(d) + kᵀ(R·Sigma·Rᵀ)k / (2d)   [covariance opt]
  4. prob = softmax(logit) over keys, then AVERAGE prob across the query heads
     of each KV group (one score per KV head).
  5. score = (prob + epsilon) * ||value||                        [vnorm opt]

The per-query-head statistics + group-averaged scores (NOT a single pooled
per-KV-head distribution) are essential to match the reference; see kvpress's
``repeat_kv`` + ``scores.view(...).mean(dim=2)``.

tangram adaptations (all faithful within the chunk-based constraint):
* The qk hook supplies POST-RoPE query/key/value (the reference recomputes
  pre-RoPE queries from hidden_states). We recover the pre-RoPE query by
  UN-ROTATING the post-RoPE query at its true global position — exact, since
  RoPE is orthogonal, and it also preserves the model's q-norm (applied before
  RoPE), so it is more faithful than re-projecting from hidden_states.
* Instead of rotating ``mu``/``Sigma`` by ``R`` (which needs a dense [d,d]
  conjugation of the covariance), we rotate the KEYS by ``Rᵀ`` — algebraically
  identical because ``(R·mu)·k == mu·(Rᵀ·k)`` and
  ``kᵀ(R·Sigma·Rᵀ)k == (Rᵀk)ᵀ·Sigma·(Rᵀk)``.
* "Observed queries" are the current chunk's queries (prior chunks are paged
  out); within the chunk every query is used (no observation window), matching
  the reference's full-sequence mean, except the first few are dropped as
  outliers exactly as the reference does (chunk-relative, ``_QUERY_OUTLIER_SINK``).

RoPE access: the qk hook passes the OUTER attention block as ``module``; its
``module.rotary_emb`` (a vLLM ``RotaryEmbedding``) provides ``cos_sin_cache``,
``rotary_dim``, and ``is_neox_style``. Models without that standard rotary
(e.g. mRoPE / deepseek-scaling) are out of scope for this scorer.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.common import (
    apply_rotary_emb_torch,
)

logger = init_logger(__name__)


class ExpectedAttentionScorer(nn.Module):
    """One (stateless) instance shared across all compressible layers.

    Input:  ``query [T, num_kv_heads * num_q_per_kv * head_size]``,
            ``key   [T, num_kv_heads * head_size]``,
            ``value [T, num_kv_heads * head_size]`` (post-RoPE, token-major),
            plus ``module`` (outer attention block, owns ``rotary_emb``) and
            ``position_offset`` (chunk's global start position).
    Output: scores ``[num_kv_heads, T]`` (float32), higher = more important.
    """

    consumes = "qk"
    name = "expected_attention"

    #: First queries of each chunk dropped from the mean/covariance estimate as
    #: outliers (kvpress ``get_query_statistics`` hardcodes 4; chunk-relative).
    _QUERY_OUTLIER_SINK = 4

    def __init__(
        self,
        num_kv_heads: int,
        num_q_per_kv: int,
        head_size: int,
        use_covariance: bool = True,
        use_vnorm: bool = True,
        n_future_positions: int = 512,
        epsilon: float = 1e-2,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.num_q_per_kv = num_q_per_kv
        self.head_size = head_size
        self.use_covariance = use_covariance
        self.use_vnorm = use_vnorm
        self.n_future_positions = n_future_positions
        self.epsilon = epsilon

    def _cos_sin(
        self, rotary_emb: nn.Module, positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` (float32, ``[len(positions), rotary_dim // 2]``)
        from the model's rotary cache at the given absolute positions."""
        cache = rotary_emb.cos_sin_cache.to(
            device=positions.device, dtype=torch.float32)
        max_pos = cache.shape[0]
        # Safety net: positions past the cache collapse onto the last entry.
        # Callers warn when this is reachable (see the future-position site).
        positions = positions.clamp_max(max_pos - 1)
        cos, sin = cache.index_select(0, positions).chunk(2, dim=-1)
        return cos, sin

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
        if module is None or not hasattr(module, "rotary_emb"):
            raise RuntimeError(
                "ExpectedAttentionScorer needs the outer attention block (with "
                "rotary_emb) as `module`; got "
                f"{type(module).__name__ if module is not None else None}.")
        rotary_emb = module.rotary_emb
        rotary_dim = rotary_emb.rotary_dim
        is_neox = rotary_emb.is_neox_style

        n_kv = self.num_kv_heads
        groups = self.num_q_per_kv
        d = self.head_size
        n_q = n_kv * groups
        T = query.shape[0]

        # float32 throughout for a stable mean / covariance / softmax.
        q = query.reshape(T, n_q, d).float()
        k = key.reshape(T, n_kv, d).float()

        positions = torch.arange(
            position_offset, position_offset + T, device=query.device)

        # --- 1. recover pre-RoPE queries by un-rotating at true positions ---
        cos, sin = self._cos_sin(rotary_emb, positions)          # [T, rd/2]
        q_pre = self._rotate(q, cos, -sin, rotary_dim, is_neox)  # inverse RoPE

        # Drop the first few chunk queries as outliers (kvpress, chunk-relative).
        sink_q = min(self._QUERY_OUTLIER_SINK, T - 1)
        q_obs = q_pre[sink_q:]                                   # [To, n_q, d]
        n_obs = q_obs.shape[0]

        # --- 2. PER-QUERY-HEAD mean / covariance of the pre-RoPE queries ---
        mu = q_obs.mean(dim=0)                                   # [n_q, d]
        use_cov = self.use_covariance and n_obs >= 2
        if use_cov:
            centered = q_obs - mu                                # [To, n_q, d]
            # Covariance normalised by the number of query positions (kvpress).
            cov = torch.einsum(
                "sni,snj->nij", centered, centered) / n_obs      # [n_q, d, d]

        # --- 3. average future-position rotation, applied to the keys as Rᵀ ---
        seq_end = position_offset + T
        # The look-ahead window can run past the rotary cache near the model's
        # max position; _cos_sin then clamps it, making the averaged future
        # rotation approximate for the tail. Warn once (computed from ints, no
        # device sync).
        rope_max = rotary_emb.cos_sin_cache.shape[0]
        if seq_end + self.n_future_positions > rope_max:
            logger.warning_once(
                "ExpectedAttention: look-ahead positions exceed the rotary "
                "cache (%d); clamping to the last entry, so the future "
                "rotation is approximate near the context tail.", rope_max)
        future = torch.arange(
            seq_end, seq_end + self.n_future_positions, device=query.device)
        fcos, fsin = self._cos_sin(rotary_emb, future)           # [F, rd/2]
        mcos = fcos.mean(dim=0, keepdim=True).expand(T, -1)      # [T, rd/2]
        msin = fsin.mean(dim=0, keepdim=True).expand(T, -1)
        # Rᵀ rotates by the negated mean angle (transpose of the mean rotation).
        k_rot = self._rotate(k, mcos, -msin, rotary_dim, is_neox)  # [T, n_kv, d]
        # Repeat each KV head across its query group (kvpress repeat_kv).
        k_rep = k_rot.unsqueeze(2).expand(T, n_kv, groups, d).reshape(T, n_q, d)

        # --- expected attention logit per query head: mean (+ covariance) ---
        logit = torch.einsum(
            "hd,thd->ht", mu, k_rep) / math.sqrt(d)              # [n_q, T]
        if use_cov:
            logit = logit + torch.einsum(
                "thd,hde,the->ht", k_rep, cov, k_rep) / d / 2.0

        # --- 4. softmax over keys, then average over each GQA group ---
        prob = torch.softmax(logit, dim=-1)                      # [n_q, T]
        prob = prob.reshape(n_kv, groups, T).mean(dim=1)         # [n_kv, T]

        # --- 5. value-norm reweighting (per KV head) ---
        if self.use_vnorm and value is not None:
            v = value.reshape(T, n_kv, d).float()
            vnorm = v.norm(dim=-1).transpose(0, 1)               # [n_kv, T]
            score = (prob + self.epsilon) * vnorm
        else:
            score = prob
        return score.contiguous()

    @staticmethod
    def _rotate(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rotary_dim: int,
        is_neox: bool,
    ) -> torch.Tensor:
        """Apply a RoPE rotation given by ``(cos, sin)`` to the leading
        ``rotary_dim`` channels of ``x [T, H, head_size]`` and pass the rest
        through unchanged (handles partial rotary)."""
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_rot = apply_rotary_emb_torch(x_rot, cos, sin, is_neox)
        if x_pass.shape[-1] == 0:
            return x_rot
        return torch.cat((x_rot, x_pass), dim=-1)
