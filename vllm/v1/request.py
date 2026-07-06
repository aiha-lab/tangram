# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import time
from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        eos_token_id: int | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: Optional["LoRARequest"] = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
        multi_turn_token_ids: list[list[int]] | None = None,
        turn_max_tokens: list[int] | None = None,
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        self.sampling_params = sampling_params
        self.pooling_params = pooling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.structured_output_request = StructuredOutputRequest.from_sampling_params(
            sampling_params
        )
        self.arrival_time = arrival_time if arrival_time is not None else time.time()

        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_FSM

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        # Multi-turn: ``multi_turn_token_ids[0]`` is turn 0's prompt and
        # must be a mutable list so ``advance_to_next_turn`` can extend
        # it in place. The caller ensures it matches ``prompt_token_ids``.
        if multi_turn_token_ids is not None:
            assert len(multi_turn_token_ids) >= 1, (
                "multi_turn_token_ids must contain at least one turn."
            )
            assert prompt_embeds is None, (
                "multi-turn requests cannot use prompt_embeds; pass token "
                "ids via multi_turn_token_ids instead."
            )
            self.prompt_token_ids = list(multi_turn_token_ids[0])
        else:
            self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            self.prompt_token_ids, prompt_embeds
        )
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = (
            self.prompt_token_ids.copy()
            if self.prompt_token_ids is not None
            else [0] * self.num_prompt_tokens
        )
        self.num_output_placeholders = 0  # Used in async scheduling.
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []
        self.num_encoder_inputs = len(self.mm_features)
        self.has_encoder_inputs = self.num_encoder_inputs > 0

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers
        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of requests being preempted by the scheduler
        self.num_preemptions = 0

        # The number of tokens that have been computed remotely.
        self.num_external_computed_tokens = 0

        self.block_hashes: list[BlockHash] = []
        self.get_hash_new_full_blocks: Callable[[], list[BlockHash]] | None = None
        if block_hasher is not None:
            self.get_hash_new_full_blocks = partial(block_hasher, self)
            self.block_hashes = self.get_hash_new_full_blocks()

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

        # Scheduler-side mirror of post-compression cache occupancy. Set
        # to the kept length after each compression-active prefill step,
        # then incremented by ``num_scheduled`` on non-compression steps.
        # The scheduler uses this instead of ``num_computed_tokens`` to
        # size ``allocate_slots``. ``None`` for non-compression requests.
        self.compress_max_eff_seq_len: int | None = None

        # Once-only compression: True (sticky) once the initial prefill
        # cycle finishes; the scheduler then stops emitting compression
        # metadata for this request.
        self.compression_done: bool = False

        # Multi-turn auto-advance. When ``multi_turn_token_ids`` is set,
        # it carries the per-turn user-input tokens for the whole
        # conversation; ``multi_turn_token_ids[0]`` must equal
        # ``prompt_token_ids``. ``None`` keeps the base single-turn path.
        self.multi_turn_token_ids: list[list[int]] | None = multi_turn_token_ids
        self.turn_max_tokens: list[int] | None = turn_max_tokens
        self.current_turn: int = 0
        self.num_turns: int = (
            len(multi_turn_token_ids) if multi_turn_token_ids is not None else 1
        )
        # Filled by ``advance_to_next_turn``; the scheduler appends the
        # final turn's output before freeing the request.
        self.turn_output_token_ids: list[list[int]] = []
        # Set by ``advance_to_next_turn``; scheduler reads + clears.
        self.just_advanced_to_next_turn: bool = False
        # Turn 0's ``max_tokens`` override; later turns are applied on
        # advance.
        if (
            turn_max_tokens is not None
            and multi_turn_token_ids is not None
            and pooling_params is None
        ):
            assert len(turn_max_tokens) == len(multi_turn_token_ids), (
                "turn_max_tokens must have the same length as "
                "multi_turn_token_ids."
            )
            self.max_tokens = turn_max_tokens[0]

    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            priority=request.priority,
            trace_headers=request.trace_headers,
            block_hasher=block_hasher,
            multi_turn_token_ids=request.multi_turn_token_ids,
            turn_max_tokens=request.turn_max_tokens,
        )

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        if self.get_hash_new_full_blocks is not None:
            self.block_hashes.extend(self.get_hash_new_full_blocks())

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def absorb_output_into_prompt(self) -> None:
        """Promote sampled output tokens into the prompt prefix.

        Used at multi-turn transitions (next turn's prefill must reattend
        the previous turn's output) and on preempt under compression
        (cache is gone; already-sampled tokens are re-prefilled).
        Clears ``_output_token_ids`` and extends ``prompt_token_ids``;
        ``_all_token_ids`` already contains the output tokens and stays
        unchanged.
        """
        if not self._output_token_ids:
            return
        assert self.prompt_token_ids is not None
        self.prompt_token_ids.extend(self._output_token_ids)
        self._output_token_ids.clear()
        self.num_prompt_tokens = len(self.prompt_token_ids)

    def reset_for_compression_preempt(self) -> None:
        """Rewind request state after preemption discards the KV cache under
        once-only compression.

        The resumed prefill must run as a fresh first compression cycle, so
        the compression bookkeeping is cleared and the already-sampled output
        tokens are folded into the prompt for reattention. The output budget
        (``max_tokens`` / ``min_tokens``) is reduced by the folded token count
        so that ``num_prompt_tokens + max_tokens`` stays constant across
        preemptions; without this, repeated preemption would grow that sum
        past ``max_model_len``, leaving the request permanently unschedulable
        yet unfinished (engine livelock).
        """
        self.compression_done = False
        self.compress_max_eff_seq_len = None
        num_absorbed = self.num_output_tokens
        self.absorb_output_into_prompt()
        self.max_tokens = max(1, self.max_tokens - num_absorbed)
        if self.sampling_params is not None:
            self.sampling_params.min_tokens = max(
                0, self.sampling_params.min_tokens - num_absorbed
            )

    def advance_to_next_turn(self) -> bool:
        """Multi-turn auto-advance. Called from ``check_stop`` when the
        current turn's stop conditions fire.

        Snapshots the just-finished turn's output, absorbs it into
        ``prompt_token_ids``, appends the next turn's user tokens, and
        updates ``current_turn`` / ``max_tokens``. The KV cache stays
        live so the next step resumes chunked prefill of the appended
        tokens. Returns False on the final turn (caller should finish).
        """
        if self.multi_turn_token_ids is None:
            return False
        if self.current_turn >= self.num_turns - 1:
            return False
        self.turn_output_token_ids.append(list(self._output_token_ids))
        # Full conversation history is retained so output is byte-equal
        # to base vLLM running the concatenated prompt in one shot.
        self.absorb_output_into_prompt()
        self.current_turn += 1
        next_user_tokens = self.multi_turn_token_ids[self.current_turn]
        self.prompt_token_ids.extend(next_user_tokens)
        # ``_all_token_ids`` already has the just-finished turn's output;
        # only append the new user tokens.
        self._all_token_ids.extend(next_user_tokens)
        self.num_prompt_tokens = len(self.prompt_token_ids)
        if self.turn_max_tokens is not None:
            self.max_tokens = self.turn_max_tokens[self.current_turn]
        # Scheduler reads + clears this on the next step.
        self.just_advanced_to_next_turn = True
        return True

    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        num_tokens = self.mm_features[input_id].mm_position.length
        return num_tokens

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    def __str__(self):
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
}
