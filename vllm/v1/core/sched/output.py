# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

from typing_extensions import deprecated

from vllm._bc_linter import bc_linter_include

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request
else:
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    LoRARequest = object
    MultiModalFeatureSpec = object
    PoolingParams = object
    SamplingParams = object
    Request = object


@bc_linter_include
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: "torch.Tensor | None" = None

    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prefill_token_ids=prefill_token_ids,
        )

    def __repr__(self) -> str:
        prompt_embeds_shape = self.prompt_embeds.shape if self.prompt_embeds else None
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )

    # Version of __repr__ with the prompt data obfuscated
    def anon_repr(self) -> str:
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None
        )
        prompt_embeds_shape = self.prompt_embeds.shape if self.prompt_embeds else None
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )


@bc_linter_include
@dataclass
class CachedRequestData:
    req_ids: list[str]
    # For request ids not in resumed_req_ids, new_block_ids will be appended to
    # the request's block IDs. For those in the set, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
    resumed_req_ids: set[str]
    # Multi-turn: ids of requests whose ``advance_to_next_turn`` fired this
    # step. The full updated ``all_token_ids`` is sent in ``all_token_ids``
    # and the worker refreshes its CachedRequestState (drop
    # output_token_ids, extend prompt_token_ids, update num_tokens).
    advanced_turn_req_ids: set[str]
    # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.
    # When PP is not used, new_token_ids will be empty.
    new_token_ids: list[list[int]]
    # For requests not scheduled in the last step, propagate the token ids to the
    # connector. Won't contain requests that were scheduled in the prior step.
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @cached_property
    @deprecated("use resumed_req_ids field")
    def resumed_from_preemption(self) -> list[bool]:
        return [req_id in self.resumed_req_ids for req_id in self.req_ids]

    @cached_property
    @deprecated("use all_token_ids field")
    def resumed_req_token_ids(self) -> list[list[int] | None]:
        return [
            self.all_token_ids[req_id] if req_id in self.resumed_req_ids else None
            for req_id in self.req_ids
        ]

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            advanced_turn_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


# PREFILL ASSUMPTIONS
#
# Compression runs on chunked prefill only, and extending it to decode is a
# committed direction. These are the six places that assume the step being
# compressed is a prefill chunk, collected here so the cost of that extension
# can be read in one place instead of rediscovered site by site. Pointers are
# by symbol, not by line: line numbers rot within a commit or two.
#
#   scheduler.py  Scheduler._maybe_add_compression_metadata
#       Attaches no metadata at all once num_computed_tokens reaches
#       num_prompt_tokens, so a decode step arrives with no directives.
#   scheduler.py  Scheduler._compression_chunk_cap
#       Caps a step at the distance to the next chunk boundary only while the
#       request is still inside its prompt; a decode step is never capped.
#   this dataclass  chunk_in_sequence_idx, is_last_chunk,
#                   compression_chunk_len, total_prompt_tokens
#       Four fields expressing a geometry a decode step does not have: it is
#       one token, with no chunk index, no final chunk, and no prompt total.
#   compressor.py  KVCompressor._assert_once_only
#       Requires the incoming lengths to equal what the last eviction
#       committed, which a decode step interleaved between two chunks would
#       have advanced.
#   eviction_regime.py  RatioRegime._adjusted_ratio
#       Divides by total_prompt_tokens to window-correct the keep fraction;
#       with no prompt total the fraction is undefined.
#   eviction_regime.py  _ChunkLocalWorkspace.build_eval_scores
#       Hands this chunk's observation window to the next chunk only when
#       chunk_len >= window_size. A decode step is one token, so any window
#       wider than that takes the degenerate branch and reuses a stale carry.
#
# Consequence for anything added to this dataclass: a new required field must
# not be chunk-shaped, or it adds a seventh entry to this list.


@dataclass
class CompressionRequestMetadata:
    """Per-request compression directives attached to ``SchedulerOutput``.

    Populated only for prefill chunks of compression-enabled requests;
    empty otherwise. Consumed by
    ``CompressionModelRunnerMixin._compression_step``.
    """
    req_id: str
    # Keep fraction of the re-eval region per chunk
    # (``CacheConfig.compression_keep_ratio``). Per-chunk K_new is
    # ``floor(keep_ratio * re_eval_size)``. 0 < keep_ratio <= 1; 1.0 when a
    # budget is the retention target instead.
    compression_keep_ratio: float
    window_size: int
    n_sink_tokens: int
    # Absolute per-(layer, group) ``kept_lengths`` floor; 0 disables it.
    floor_min: int
    chunk_in_sequence_idx: int
    is_last_chunk: bool
    # When True, this step closes a full ``compression_chunk_size`` boundary
    # (or the prompt end), so the worker runs the keep-decision + eviction
    # over ``compression_chunk_len`` accumulated tokens. When False, this is a
    # budget-sliced sub-chunk: the gate still scores its tokens (accumulated
    # in the compressor) but no eviction runs. Decoupling the two keeps every
    # compression step a full ``chunk_size`` chunk — byte-equivalent to the
    # serial baseline regardless of how the scheduler sliced the prefill.
    run_compression: bool = True
    # Tokens accumulated since the last compression boundary (== the chunk the
    # worker evicts when ``run_compression`` is True): ``chunk_size`` for
    # interior chunks, the remainder for the last chunk. Differs from this
    # step's scheduled token count only when budget sharing split the chunk.
    compression_chunk_len: int = 0
    # Total prompt length of this request's first prefill cycle. The ratio
    # regime derives its per-chunk target from it; the budget regime does not
    # need it (a budget is absolute).
    total_prompt_tokens: int = 0
    # Fixed per-(layer, head group) KV token budget
    # (``CacheConfig.compression_budget_tokens``), or None when the retention
    # target is a ratio. Selects the budget eviction regime in the worker.
    budget_tokens: int | None = None
    # Budget regime only: whether the fresh chunk is an eviction candidate
    # (``CacheConfig.compression_evict_current_chunk``).
    evict_current_chunk: bool = False


@bc_linter_include
@dataclass
class SchedulerOutput:
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData]
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes,
    # we only send the diff to minimize the communication cost.
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> encoder input indices that need processing.
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # to process that the request's 0-th and 1-th images in the current step.
    scheduled_encoder_inputs: dict[str, list[int]]
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    num_common_prefix_blocks: list[int]

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str]
    # list of mm_hash strings associated with the encoder outputs to be
    # freed from the encoder cache.
    free_encoder_mm_hashes: list[str]

    # Request IDs that are preempted in this step.
    # Only used for v2 model runner.
    preempted_req_ids: set[str] | None = None

    # Whether the scheduled requests have all the output tokens they
    # need to perform grammar bitmask computation.
    pending_structured_output_tokens: bool = False

    # KV Cache Connector metadata.
    kv_connector_metadata: KVConnectorMetadata | None = None

    # EC Cache Connector metadata
    ec_connector_metadata: ECConnectorMetadata | None = None

    # Per-request compression directives; empty on non-compression steps.
    compression_metadata: dict[str, CompressionRequestMetadata] = field(
        default_factory=dict
    )

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )


@dataclass
class GrammarOutput:
    # ids of structured output requests.
    structured_output_request_ids: list[str]
    # Bitmask ordered as structured_output_request_ids.
    grammar_bitmask: "npt.NDArray[np.int32]"
