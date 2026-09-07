# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate-capture custom op for FastKVZip compression scoring.

The gate scores an attention block's INPUT hidden_states, which no hook can
see: torch.compile inlines the module forward and skips its pre-hooks.
``_wrap_forward_with_gate_capture`` therefore overrides the parent block's
instance-level ``forward`` -- which dynamo does trace -- to call
``vllm::tangram_gate_capture`` on those hidden_states first, so the call
survives as an opaque graph node. The op is registered here as a
piecewise-SPLITTING op, so its side effect runs eagerly between CUDA-graph
pieces every step; it MUST stay in ``CompilationConfig._attention_ops`` or the
capture is silently dropped.

``KVCompressor.attach_scorers`` wires each layer's capture fn and calls the
wrapper; importing this module from the compressor registers the op.
"""
from __future__ import annotations

import inspect

import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op


def _wrap_forward_with_gate_capture(parent: nn.Module, layer_name: str) -> None:
    """Install a ``vllm::tangram_gate_capture`` call in front of ``parent``'s
    forward via an instance-level forward override.

    A pre-hook cannot work: torch.compile inlines module forwards and skips
    their hooks. An instance-level ``forward`` IS traced, because
    ``nn.Module.__call__`` resolves ``self.forward`` through the instance, and
    the op call inside becomes an opaque node piecewise compilation splits on,
    so the capture body runs eagerly every step.

    ``layer_name`` identifies the inner ``Attention`` the op body reads the
    capture fn off. The hidden_states argument position is resolved once here,
    since attention-block forwards differ across models.
    """
    if getattr(parent, "_tangram_gate_capture_wrapped", False):
        return
    orig_forward = parent.forward
    params = list(inspect.signature(orig_forward).parameters)
    try:
        hs_index = params.index("hidden_states")
    except ValueError as exc:
        raise RuntimeError(
            f"Gate capture: attention block {type(parent).__name__} has no "
            f"'hidden_states' parameter (found {params}); cannot deliver "
            "hidden states to the FastKVZip gate scorer."
        ) from exc

    def forward_with_gate_capture(*args, **kwargs):
        if "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
        else:
            hidden_states = args[hs_index]
        torch.ops.vllm.tangram_gate_capture(hidden_states, layer_name)
        return orig_forward(*args, **kwargs)

    parent.forward = forward_with_gate_capture
    parent._tangram_gate_capture_wrapped = True


def tangram_gate_capture(hidden_states: torch.Tensor, layer_name: str) -> None:
    """Deliver an attention block's input hidden_states to the compression
    gate scorer (FastKVZip).

    INVARIANT: must remain in ``CompilationConfig._attention_ops``, the
    piecewise splitting ops. Its whole purpose is the Python side effect of
    scoring and stashing, and a splitting op runs eagerly between CUDA-graph
    pieces every step -- one captured inside a graph would run at capture time
    only and be silently skipped on every replay.

    Declared as mutating ``hidden_states``, which it never writes, so the schema
    neither aliases input to output nor lets the node be dropped as dead code.
    ``vllm::maybe_calc_kv_scales`` uses the same pattern.
    """
    forward_context = get_forward_context()
    capture = forward_context.no_compile_layers[layer_name].compression_gate_capture
    if capture is not None:
        capture(hidden_states)


def tangram_gate_capture_fake(
    hidden_states: torch.Tensor, layer_name: str
) -> None:
    return


direct_register_custom_op(
    op_name="tangram_gate_capture",
    op_func=tangram_gate_capture,
    mutates_args=["hidden_states"],
    fake_impl=tangram_gate_capture_fake,
)
