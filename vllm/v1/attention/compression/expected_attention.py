# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ExpectedAttention scorer — analytic expected-attention score (axis 2).

Estimates the attention each key will draw from FUTURE decode queries without
materialising an attention matrix. Ported from NVIDIA KVpress
(``expected_attention_press.py``). Per query head, then averaged over the GQA
group -- per-head statistics with the group average taken LAST, not one pooled
per-KV-head distribution, is what matches the reference::

    mu, cov   = mean and covariance of the PRE-RoPE chunk queries
    R         = average RoPE rotation of the next n_future_positions
    logit(k)  = (R·mu)·k / sqrt(d) + kᵀ(R·cov·Rᵀ)k / (2d)
    score(k)  = (softmax_k(logit) + eps) * ||v||

Two algebraic identities keep this exact on post-RoPE inputs: the pre-RoPE query
is recovered by un-rotating at the true global position (RoPE is orthogonal, and
this preserves the model's pre-RoPE q-norm), and the keys are rotated by ``Rᵀ``
rather than conjugating the covariance, since ``(R·mu)·k == mu·(Rᵀ·k)`` and
``kᵀ(R·cov·Rᵀ)k == (Rᵀk)ᵀ·cov·(Rᵀk)``.

Needs a standard ``module.rotary_emb``; mRoPE and deepseek-scaling are out of
scope.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from vllm.v1.attention.compression.qk_scorer_base import QKScorer
from vllm.v1.attention.compression.scorer_options import ScorerOption
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.common import (
    apply_rotary_emb_torch,
)

logger = init_logger(__name__)


class ExpectedAttentionScorer(QKScorer):
    """Needs every argument of the shared contract: ``value`` for the norm
    reweighting, ``module`` for its ``rotary_emb``, and ``position_offset`` to
    un-rotate the queries at their true global positions."""

    name = "expected_attention"

    #: First queries of each chunk dropped from the mean/covariance estimate as
    #: outliers (kvpress ``get_query_statistics`` hardcodes 4; chunk-relative).
    _QUERY_OUTLIER_SINK = 4

    OPTIONS = (
        ScorerOption(
            "use_covariance", bool, True,
            "Add the query covariance term to the expected attention logit "
            "(kvpress default)."),
        ScorerOption(
            "use_vnorm", bool, True,
            "Reweight the expected attention by the value norm (kvpress "
            "default)."),
        ScorerOption(
            "n_future_positions", int, 512,
            "Number of future decode positions whose RoPE rotation is averaged "
            "to anticipate where later queries attend.",
            requirement=("a positive integer", lambda v: v > 0)),
        ScorerOption(
            "epsilon", float, 1e-2,
            "Constant added before the value-norm reweighting, bounding the "
            "score of a near-zero-norm value.",
            requirement=("a non-negative float", lambda v: v >= 0)),
    )

    def __init__(
        self,
        num_kv_heads: int,
        head_size: int,
        num_q_per_kv: int = 1,
        *,
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
        # Near the model's max position the look-ahead runs past the rotary
        # cache and _cos_sin clamps it, so the averaged future rotation is
        # approximate for the tail. Warned once, from ints, no device sync.
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
