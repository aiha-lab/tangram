# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compression gate module — per-layer importance score MLP.

Architecture (Fast-KVzip gate): q_proj/k_proj with weight-only RMSNorm,
sink-anchor ``k_base``, per-head bias ``b``; score is a sigmoid-like over
(logit − logit_base) averaged across groups. Trained checkpoints are
downloaded from the ``hmkim97/tangram-gate`` Hub repo."""
from __future__ import annotations

import math
import os
import re

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

# Hub repo hosting the trained gate checkpoints, laid out as per-model
# subdirectories (e.g. ``qwen3-4b-instruct-2507/q4_dim16_sink16.pt``).
_GATE_HF_REPO = "hmkim97/tangram-gate"
# Pin to an immutable commit so a later push to the repo can't silently swap
# the weights behind an unchanged filename (reproducibility).
_GATE_HF_REVISION = "06a628fab229ff3075f69b61f43fa0ef6631c875"
_GATE_SHORT_ID_ALIASES = {
    "llama-3.1-8b-instruct": "llama3.1-8b-instruct",
}


class _RMSNorm(nn.Module):
    # eps=1e-6 matches Qwen3RMSNorm so official gate checkpoints load
    # bit-identically with no key remapping.
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * x.to(in_dtype)


class CompressionGate(nn.Module):
    """One per transformer layer.

    Input:  hidden_states ``[T, hidden_dim]`` or ``[1, T, hidden_dim]``.
    Output: scores ``[num_kv_heads, T]``.
    """

    # Axis-2 dispatch: the gate consumes the outer attention block's
    # hidden_states.
    consumes = "hidden_states"
    name = "fastkvzip"

    def __init__(
        self,
        layer_idx: int,
        hidden_dim: int,
        output_dim: int,
        num_kv_heads: int,
        num_groups: int,
        sink_dim: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_groups
        self.sink_dim = sink_dim
        self._scale = math.sqrt(output_dim)

        self.q_proj = nn.Linear(
            hidden_dim,
            num_kv_heads * num_groups * output_dim,
            bias=True,
            dtype=dtype,
        )
        self.k_proj = nn.Linear(
            hidden_dim,
            num_kv_heads * output_dim,
            bias=False,
            dtype=dtype,
        )
        self.q_norm = _RMSNorm(output_dim)
        self.k_norm = _RMSNorm(output_dim)
        self.k_base = nn.Parameter(
            torch.zeros(num_kv_heads, 1, sink_dim, output_dim))
        self.b = nn.Parameter(
            torch.zeros(num_kv_heads, 1, num_groups, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3:
            assert hidden_states.size(0) == 1, (
                "CompressionGate expects unbatched [T, D] or [1, T, D]; got "
                f"batch={hidden_states.size(0)}")
            hidden_states = hidden_states.squeeze(0)

        T = hidden_states.size(0)
        num_kv_heads = self.num_kv_heads
        num_groups = self.num_groups
        output_dim = self.output_dim

        q = self.q_norm(self.q_proj(hidden_states).view(
            T, num_kv_heads, num_groups, output_dim))
        k = self.k_norm(self.k_proj(hidden_states).view(
            T, num_kv_heads, 1, output_dim))

        # [T, num_kv_heads, num_groups, output_dim]
        # → [num_kv_heads, T, output_dim, num_groups]
        q = q.transpose(0, 1).transpose(-1, -2)
        # [T, num_kv_heads, 1, output_dim] → [num_kv_heads, T, 1, output_dim]
        k = k.transpose(0, 1)

        # logit:      [num_kv_heads, T, 1, num_groups]
        logit = torch.matmul(k, q) / self._scale + self.b.unsqueeze(2)
        # logit_base: [num_kv_heads, T, sink, num_groups]
        logit_base = torch.matmul(self.k_base, q) / self._scale

        # 1 / (1 + sum_sink(exp(logit_base - logit))) — sigmoid-like over
        # the anchor set; equals sigmoid when sink_dim == 1.
        score = 1.0 / (
            1.0 + torch.exp(logit_base - logit).sum(2, keepdim=True))
        score = score.mean(-1)
        return score.squeeze(-1)


def _resolve_gate_filename(model_name: str, gate_path: str) -> str:
    # "fastkvzip" → derive "{model_short}/q{num_groups}_dim16_sink16.pt"
    # from the model's GQA ratio, matching the tangram-gate upload layout.
    if gate_path != "fastkvzip":
        return gate_path
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_name)
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    num_groups = cfg.num_attention_heads // cfg.num_key_value_heads
    fname = f"q{num_groups}_dim16_sink16"
    short = model_name.rstrip("/").split("/")[-1].lower()
    short = _GATE_SHORT_ID_ALIASES.get(short, short)
    return os.path.join(short, fname + ".pt")


def _local_gate_path(resolved: str) -> str | None:
    # Local Fast-KVzip output dir, checked before any network so air-gapped
    # deployments work. None when absent.
    candidate = os.path.expanduser(
        os.path.join("~", "FastKVzip", "result_gate", resolved))
    return candidate if os.path.exists(candidate) else None


def _hf_cached_gate_path(resolved: str) -> str | None:
    # HF cache lookup only, no network (local_files_only); None on a miss.
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (EntryNotFoundError,
                                       LocalEntryNotFoundError)
    try:
        return hf_hub_download(
            repo_id=_GATE_HF_REPO,
            filename=resolved,
            repo_type="model",
            revision=_GATE_HF_REVISION,
            local_files_only=True,
        )
    except (LocalEntryNotFoundError, EntryNotFoundError):
        return None


def _hf_download_gate_path(resolved: str, gate_path: str) -> str:
    # Download from the pinned revision; on failure raise with the cause
    # distinguished instead of a generic "not found".
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (EntryNotFoundError, GatedRepoError,
                                       HfHubHTTPError, OfflineModeIsEnabled,
                                       RepositoryNotFoundError,
                                       RevisionNotFoundError)
    try:
        return hf_hub_download(
            repo_id=_GATE_HF_REPO,
            filename=resolved,
            repo_type="model",
            revision=_GATE_HF_REVISION,
        )
    except (RepositoryNotFoundError, GatedRepoError) as e:
        raise RuntimeError(
            f"Compression gate repo '{_GATE_HF_REPO}' is inaccessible "
            f"(deleted, made private, or the HF token is missing/"
            f"unauthorized). Set a valid HF token, or stage the checkpoint "
            f"locally so no Hub access is needed.") from e
    except RevisionNotFoundError as e:
        raise RuntimeError(
            f"Compression gate repo '{_GATE_HF_REPO}' has no revision "
            f"'{_GATE_HF_REVISION}' (the pinned commit was rewritten or "
            f"removed).") from e
    except EntryNotFoundError as e:
        raise FileNotFoundError(
            f"Compression gate checkpoint '{gate_path}' (resolved to "
            f"'{resolved}') does not exist in '{_GATE_HF_REPO}' at revision "
            f"'{_GATE_HF_REVISION}'.") from e
    except (OfflineModeIsEnabled, HfHubHTTPError) as e:
        raise RuntimeError(
            f"Could not reach the compression gate repo '{_GATE_HF_REPO}' "
            f"(offline mode or network error) and '{resolved}' is not in the "
            f"local cache. Stage the checkpoint locally for offline use.") \
            from e


def _download_or_local(model_name: str, gate_path: str) -> str:
    # Try every offline source before the network, in order:
    # (1) absolute path, (2) local FastKVzip dir, (3) HF cache, (4) download.
    if os.path.isabs(gate_path) and os.path.exists(gate_path):
        return gate_path
    resolved = _resolve_gate_filename(model_name, gate_path)
    return (_local_gate_path(resolved)
            or _hf_cached_gate_path(resolved)
            or _hf_download_gate_path(resolved, gate_path))


def _shard_gate_state_dict(
    state_dict: dict,
    num_heads_total: int,
    num_heads_per_rank: int,
    tp_rank: int,
) -> dict:
    """Slice a gate state_dict along the KV-head axis for one TP rank.

    Matches QKVParallelLinear's contiguous head shard: rank ``r`` of
    ``N`` takes heads ``[r * per_rank, (r + 1) * per_rank)``. No-op when
    total == per_rank. Norm weights are kv-head-independent.
    """
    if num_heads_total == num_heads_per_rank:
        return state_dict
    if num_heads_total % num_heads_per_rank != 0:
        raise ValueError(
            f"gate shard: num_heads_total ({num_heads_total}) must be a "
            f"multiple of num_heads_per_rank ({num_heads_per_rank}).")
    head_start = tp_rank * num_heads_per_rank
    head_end = head_start + num_heads_per_rank
    out: dict = {}
    for key, value in state_dict.items():
        if key == "q_proj.weight":
            inner = value.shape[0] // num_heads_total
            value = value.reshape(
                num_heads_total, inner, value.shape[1])[head_start:head_end]
            value = value.reshape(num_heads_per_rank * inner, -1)
        elif key == "q_proj.bias":
            inner = value.shape[0] // num_heads_total
            value = value.reshape(
                num_heads_total, inner)[head_start:head_end].reshape(
                    num_heads_per_rank * inner)
        elif key == "k_proj.weight":
            output_dim = value.shape[0] // num_heads_total
            value = value.reshape(
                num_heads_total, output_dim,
                value.shape[1])[head_start:head_end]
            value = value.reshape(num_heads_per_rank * output_dim, -1)
        elif key == "k_proj.bias":
            output_dim = value.shape[0] // num_heads_total
            value = value.reshape(
                num_heads_total, output_dim)[head_start:head_end].reshape(
                    num_heads_per_rank * output_dim)
        elif key in ("k_base", "b"):
            value = value[head_start:head_end]
        # q_norm / k_norm weights are output_dim only — pass through.
        out[key] = value
    return out


def load_gates(
    model_name: str,
    gate_path: str,
    num_layers: int,
    num_kv_heads_per_rank: int,
    num_kv_heads_total: int,
    tp_rank: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> list[CompressionGate]:
    """Load per-layer gate modules from a Fast-KVzip checkpoint.

    Shapes (``num_groups``, ``output_dim``, ``sink_dim``) are inferred
    from the layer-0 state dict. Under TP, the checkpoint stores
    global KV heads; each rank slices its own range via
    ``_shard_gate_state_dict``. Errors out (no random-init fallback) if
    the checkpoint is missing or shape-mismatched.
    """
    file_path = _download_or_local(model_name, gate_path)
    # full unpickling is trusted here. Only blob["module"] (tensors) is consumed.
    blob = torch.load(file_path, weights_only=False, map_location="cpu")
    state_dicts: list[dict] = blob["module"]
    if len(state_dicts) != num_layers:
        raise ValueError(
            f"Compression gate '{gate_path}' has {len(state_dicts)} layers "
            f"but model has {num_layers}.")

    # Infer dims from layer 0; all layers share the same shape.
    layer0 = state_dicts[0]
    # [num_kv_heads * num_groups * output_dim, hidden]
    q_out, q_in = layer0["q_proj.weight"].shape
    # [num_kv_heads * output_dim, hidden]
    k_out, _ = layer0["k_proj.weight"].shape
    output_dim = layer0["q_norm.weight"].shape[-1]
    num_kv_heads_ckpt = k_out // output_dim
    num_groups = q_out // k_out
    sink_dim = layer0["k_base"].shape[2]

    if q_in != hidden_dim:
        raise ValueError(
            f"Compression gate hidden_dim mismatch: ckpt={q_in}, "
            f"model={hidden_dim}.")
    if num_kv_heads_ckpt != num_kv_heads_total:
        raise ValueError(
            f"Compression gate num_kv_heads mismatch: "
            f"ckpt={num_kv_heads_ckpt}, model.total={num_kv_heads_total}.")
    if num_kv_heads_total % num_kv_heads_per_rank != 0:
        raise ValueError(
            f"Compression gate: num_kv_heads_total ({num_kv_heads_total}) "
            f"must be a multiple of num_kv_heads_per_rank "
            f"({num_kv_heads_per_rank}).")

    # k_base's third dim is the authoritative sink count (the gate is built and
    # weight-loaded against it). A differing filename-encoded count means a
    # mislabeled checkpoint — fail loudly, like the mismatch checks above.
    m = re.search(r"sink(\d+)", os.path.basename(file_path))
    if m and int(m.group(1)) != sink_dim:
        raise ValueError(
            f"Compression gate sink_dim mismatch for '{gate_path}': "
            f"checkpoint tensor has sink_dim={sink_dim} but the filename "
            f"encodes sink_dim={int(m.group(1))}. The checkpoint is "
            f"mislabeled or wrongly paired.")

    gates: list[CompressionGate] = []
    for layer_idx, state_dict in enumerate(state_dicts):
        sharded = _shard_gate_state_dict(
            state_dict, num_kv_heads_total, num_kv_heads_per_rank, tp_rank)
        gate = CompressionGate(
            layer_idx=layer_idx,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_kv_heads=num_kv_heads_per_rank,
            num_groups=num_groups,
            sink_dim=sink_dim,
            dtype=dtype,
        )
        gate.load_state_dict(sharded)
        gate.eval()
        gates.append(gate.to(device=device))

    logger.info(
        "Loaded Compression gate '%s' (%d layers, "
        "num_kv_heads_per_rank=%d, num_kv_heads_total=%d, tp_rank=%d, "
        "num_groups=%d, output_dim=%d, sink_dim=%d).",
        gate_path, num_layers, num_kv_heads_per_rank, num_kv_heads_total,
        tp_rank, num_groups, output_dim, sink_dim,
    )
    return gates
