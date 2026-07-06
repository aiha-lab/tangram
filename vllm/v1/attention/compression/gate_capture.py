# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate-capture custom op for FastKVZip compression scoring.

The FastKVZip gate scores an attention block's *input* hidden_states, but under
torch.compile the module forward is inlined and its pre-hooks are skipped, so a
hook cannot see those hidden_states. This module solves that with a custom op:

* :func:`_wrap_forward_with_gate_capture` overrides a parent attention block's
  instance-level ``forward`` to call the ``vllm::tangram_gate_capture`` op with
  the block's input hidden_states before running the real forward. dynamo
  traces the instance forward (unlike a hook), so the op call survives as an
  opaque graph node.
* ``vllm::tangram_gate_capture`` (registered here) is a piecewise-splitting op,
  so its Python side effect — handing hidden_states to the layer's capture fn —
  runs eagerly between CUDA-graph pieces on every step. It MUST stay in
  ``CompilationConfig._attention_ops`` for this to hold.

``KVCompressor.attach_scorers`` wires each compressible layer's capture fn onto
its inner ``Attention`` and calls :func:`_wrap_forward_with_gate_capture` on the
outer block. Importing this module from the compressor is what registers the op.
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

    A forward pre-hook cannot be used: torch.compile inlines module forwards
    and skips their hooks. An instance-level ``forward`` IS traced by dynamo
    (``nn.Module.__call__`` resolves ``self.forward`` through the instance),
    and the op call inside it becomes an opaque graph node that piecewise
    compilation splits on, so the capture body runs eagerly on every step.

    ``layer_name`` identifies the layer's inner ``Attention`` in the forward
    context; the op body reads the capture fn off it. The hidden_states
    argument position is resolved once here — attention-block forwards differ
    across models (e.g. ``(positions, hidden_states)`` for Qwen/Llama).
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

    INVARIANT: this op must remain in ``CompilationConfig._attention_ops``
    (piecewise splitting ops). Its entire purpose is the Python side effect of
    scoring + stashing; a splitting op executes eagerly between CUDA-graph
    pieces on every step, whereas an op captured inside a graph would run at
    capture time only and be silently skipped on every replay.

    Declared as mutating ``hidden_states`` (it never actually writes) so the
    schema neither aliases input to output nor lets the node be eliminated as
    dead code; ``vllm::maybe_calc_kv_scales`` uses the same pattern.
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
