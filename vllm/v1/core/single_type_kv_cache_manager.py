# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    HeadGroupedAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request


class SingleTypeKVCacheManager(ABC):
    """
    An abstract base class for a manager that handle the kv cache management
    logic of one specific type of attention layer.
    """

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> None:
        """
        Initializes the SingleTypeKVCacheManager.
        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            block_pool: The block pool.
            kv_cache_group_id: The id of the kv cache group of this manager.
        """
        self.block_size = kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size * pcp_world_size > 1:
            self.block_size *= dcp_world_size * pcp_world_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool

        # Head-grouped paging: a token-position requires
        # ``num_head_groups_total = num_head_groups_per_layer × num_layers``
        # globally-unique block ids, popped in one
        # ``BlockPool.get_new_blocks`` call.
        self.head_grouped = isinstance(kv_cache_spec, HeadGroupedAttentionSpec)
        if self.head_grouped:
            self.num_head_groups_total = block_pool.num_head_groups
            assert self.num_head_groups_total is not None, (
                "BlockPool.num_head_groups must be set for "
                "HeadGroupedAttentionSpec.")
        else:
            self.num_head_groups_total = 1

        # Per-request allocated blocks. Source of truth for the non-
        # head-grouped path; left empty in head-grouped mode (see
        # ``req_to_block_ids``).
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(
            list)

        # Head-grouped only: per-request int32 ndarray of physical block
        # ids, ordered group-major within a layer (matches
        # ``BlockTable.append_row``). Source of truth here because
        # compression-driven frees update it via a numpy bitmap filter
        # in ``free_blocks_by_ids``. Block objects are materialised on
        # demand for the rare paths that need them.
        self.req_to_block_ids: dict[str, np.ndarray] = {}

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for preempted ones.
        self.num_cached_block: dict[str, int] = {}

        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock] | np.ndarray,
    ) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching. Accepts ``Sequence[KVCacheBlock]`` (base
                path) or ``np.ndarray`` of block ids (head-grouped path
                — always empty since prefix caching is disabled there).

        Returns:
            The number of blocks.
        """

        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_groups = self.num_head_groups_total
        is_ndarray = isinstance(new_computed_blocks, np.ndarray)
        num_computed = (int(new_computed_blocks.size) if is_ndarray
                        else len(new_computed_blocks))
        if self.head_grouped:
            ids = self.req_to_block_ids.get(request_id)
            existing_len = 0 if ids is None else int(ids.size)
        else:
            existing_len = len(self.req_to_blocks[request_id])
        num_new_blocks = (
            num_required_blocks * num_groups - num_computed - existing_len)
        # If a computed block is an eviction candidate (ref_cnt == 0), it
        # will be revived during allocation so it counts as new. The
        # ndarray branch carries no per-block ref_cnt (head-grouped caching
        # is disabled) and the count is always 0.
        if is_ndarray:
            num_evictable_computed_blocks = 0
        else:
            num_evictable_computed_blocks = sum(
                blk.ref_cnt == 0 and not blk.is_null for blk in new_computed_blocks
            )
        return num_new_blocks + num_evictable_computed_blocks

    def save_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock] | np.ndarray,
    ) -> None:
        """
        Add the new computed blocks to the request.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache. ``Sequence[KVCacheBlock]`` (base path) or
                ``np.ndarray`` of block ids (head-grouped path).
        """
        is_ndarray = isinstance(new_computed_blocks, np.ndarray)
        num_computed = (int(new_computed_blocks.size) if is_ndarray
                        else len(new_computed_blocks))
        if request_id not in self.num_cached_block:
            if self.head_grouped:
                assert (
                    request_id not in self.req_to_block_ids
                    or self.req_to_block_ids[request_id].size == 0
                )
                if num_computed:
                    if is_ndarray:
                        self.req_to_block_ids[request_id] = (
                            new_computed_blocks.astype(np.int32, copy=False)
                        )
                    else:
                        self.req_to_block_ids[request_id] = np.fromiter(
                            (b.block_id for b in new_computed_blocks),
                            dtype=np.int32,
                            count=num_computed,
                        )
            else:
                req_blocks = self.req_to_blocks[request_id]
                assert len(req_blocks) == 0
                req_blocks.extend(new_computed_blocks)
            self.num_cached_block[request_id] = num_computed
        else:
            # A running request must not have new computed blocks.
            assert num_computed == 0

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int
    ) -> list[KVCacheBlock] | np.ndarray:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).

        Returns:
            ``list[KVCacheBlock]`` for the base path,
            ``np.ndarray[int32]`` of block ids for the head-grouped path.
        """
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_groups = self.num_head_groups_total
        if self.head_grouped:
            existing = self.req_to_block_ids.get(request_id)
            existing_len = 0 if existing is None else int(existing.size)
            num_new_blocks = num_required_blocks * num_groups - existing_len
            if num_new_blocks <= 0:
                return np.empty(0, dtype=np.int32)
            # Pull new ids directly from the ring buffer; the int32 array
            # bypasses per-block ref_cnt updates and KVCacheBlock
            # materialisation.
            new_ids = self.block_pool.get_new_block_ids(num_new_blocks)
            if existing is None or existing.size == 0:
                self.req_to_block_ids[request_id] = new_ids
            else:
                self.req_to_block_ids[request_id] = np.concatenate(
                    [existing, new_ids]
                )
            return new_ids
        req_blocks = self.req_to_blocks[request_id]
        num_new_blocks = num_required_blocks * num_groups - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_tokens: The total number of tokens that need to be cached
                (including tokens that are already cached).
        """
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached_blocks >= num_full_blocks:
            return

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.get_req_blocks(request.request_id),
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
        )

        self.num_cached_block[request.request_id] = num_full_blocks

    def get_req_blocks(self, request_id: str) -> list[KVCacheBlock]:
        """Return the request's current ``KVCacheBlock`` list.

        Base path returns the eagerly-maintained list (mutations persist).
        Head-grouped mode materialises a fresh list from
        ``req_to_block_ids`` via ``block_pool.blocks[bid]``; mutations to
        the returned list are not visible to subsequent calls. Prefer
        ``get_req_block_ids_array`` when only ids are needed.
        """
        if self.head_grouped:
            ids = self.req_to_block_ids.get(request_id)
            if ids is None or ids.size == 0:
                return []
            pool_blocks = self.block_pool.blocks
            return [pool_blocks[int(bid)] for bid in ids.tolist()]
        return self.req_to_blocks.get(request_id, [])

    def get_req_block_ids_array(self, request_id: str) -> np.ndarray:
        """Return the request's int32 block-id ndarray. Head-grouped only.

        Returns an empty ndarray (not None) when the request has no blocks.
        """
        assert self.head_grouped, (
            "get_req_block_ids_array is head-grouped only."
        )
        ids = self.req_to_block_ids.get(request_id)
        if ids is None:
            return np.empty(0, dtype=np.int32)
        return ids

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        if self.head_grouped:
            # int32 array is the source of truth; push it back to the ring
            # buffer directly. Caching is disabled here, so eviction
            # ordering does not matter.
            ids = self.req_to_block_ids.pop(request_id, None)
            self.req_to_blocks.pop(request_id, None)
            if ids is None or ids.size == 0:
                self.num_cached_block.pop(request_id, None)
                return
            # The sliding-window eviction path (``null_blocks_by_ids``) nulls
            # out-of-window front blocks in place (sets them to the null id 0)
            # after their physical blocks were already returned to the pool.
            # Exclude the null id here so the ring is not corrupted —
            # ``free_block_ids_array`` requires null-free, unique ids. No-op when
            # no nulling occurred.
            ids = ids[ids != 0]
            if ids.size:
                self.block_pool.free_block_ids_array(ids)
            self.num_cached_block.pop(request_id, None)
            return
        # Default to [] in case a request is freed (aborted) before alloc.
        req_blocks = self.req_to_blocks.pop(request_id, [])

        # Free blocks in reverse order so that the tail blocks are
        # freed first.
        ordered_blocks = reversed(req_blocks)

        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request_id, None)

    def free_blocks_by_ids(
        self,
        block_ids: "set[int] | np.ndarray",
        candidate_req_ids: set[str] | None = None,
    ) -> None:
        """Drop named block ids from the per-request lookup after
        compression so allocations don't double-count them.

        Head-grouped mode filters the int32 ``req_to_block_ids`` array
        with a numpy bitmap; ``req_to_blocks`` is rebuilt on demand by
        ``get_req_blocks`` and is not touched here. The base path uses a
        set-membership list comprehension.

        ``block_ids`` accepts an int32/int64 ndarray (head-grouped) or a
        ``set[int]`` (base). ``candidate_req_ids`` narrows the sweep; None
        scans every running request.
        """
        is_ndarray = isinstance(block_ids, np.ndarray)
        if is_ndarray:
            if block_ids.size == 0:
                return
        elif not block_ids:
            return
        if self.head_grouped:
            num_pool_blocks = len(self.block_pool.blocks)
            if is_ndarray:
                freed_arr = block_ids.astype(np.int64, copy=False)
            else:
                freed_arr = np.fromiter(block_ids, dtype=np.int64,
                                        count=len(block_ids))
            # Drop out-of-range ids to preserve the ``not in set`` silent
            # no-op semantics of the base path.
            in_range = freed_arr < num_pool_blocks
            if not in_range.all():
                freed_arr = freed_arr[in_range]
                if freed_arr.size == 0:
                    return
            bitmap = np.zeros(num_pool_blocks, dtype=bool)
            bitmap[freed_arr] = True
            if candidate_req_ids is None:
                iter_ids = list(self.req_to_block_ids.keys())
            else:
                iter_ids = candidate_req_ids
            for req_id in iter_ids:
                ids = self.req_to_block_ids.get(req_id)
                if ids is None or ids.size == 0:
                    continue
                is_freed = bitmap[ids]
                if not is_freed.any():
                    continue
                self.req_to_block_ids[req_id] = ids[~is_freed]
            return
        # Base path. ``in`` membership is O(N) on ndarray, so convert any
        # stray ndarray to a set once.
        if is_ndarray:
            block_ids = set(block_ids.tolist())
        if candidate_req_ids is None:
            iter_ids = self.req_to_blocks.keys()
        else:
            iter_ids = candidate_req_ids
        for req_id in iter_ids:
            blocks = self.req_to_blocks.get(req_id)
            if not blocks:
                continue
            kept = [blk for blk in blocks if blk.block_id not in block_ids]
            if len(kept) != len(blocks):
                self.req_to_blocks[req_id] = kept

    def null_blocks_by_ids(
        self,
        block_ids: np.ndarray,
        candidate_req_ids: set[str] | None = None,
    ) -> None:
        """Null named block ids IN PLACE in the per-request lookup (head-grouped).

        Length-preserving counterpart of ``free_blocks_by_ids``: instead of
        dropping the freed ids (which lets the next allocation re-grow those
        slots), it overwrites them with the null block id (0) and keeps the
        array length. Used by sliding-window eviction, where the block-table row
        stays full length — only the out-of-window FRONT blocks become null, so
        the in-window tail keeps its position -> block mapping — while the
        physical blocks are returned to the pool separately (caller). The next
        ``get_num_blocks_to_allocate`` therefore counts the nulled slots as
        present and never re-allocates them.
        """
        if not self.head_grouped:
            raise AssertionError("null_blocks_by_ids is head-grouped only.")
        if block_ids.size == 0:
            return
        num_pool_blocks = len(self.block_pool.blocks)
        freed_arr = block_ids.astype(np.int64, copy=False)
        # Drop out-of-range / null (0) ids, matching the silent no-op semantics
        # of ``free_blocks_by_ids``; the null block must never be matched.
        in_range = (freed_arr > 0) & (freed_arr < num_pool_blocks)
        if not in_range.all():
            freed_arr = freed_arr[in_range]
            if freed_arr.size == 0:
                return
        bitmap = np.zeros(num_pool_blocks, dtype=bool)
        bitmap[freed_arr] = True
        if candidate_req_ids is None:
            iter_ids = list(self.req_to_block_ids.keys())
        else:
            iter_ids = candidate_req_ids
        for req_id in iter_ids:
            ids = self.req_to_block_ids.get(req_id)
            if ids is None or ids.size == 0:
                continue
            is_freed = bitmap[ids]
            if not is_freed.any():
                continue
            # In-place overwrite with the null block id; array length preserved.
            ids[is_freed] = 0

    @abstractmethod
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        Get the number of common prefix blocks for all requests with allocated
        KV cache.

        Args:
            running_request_id: The request ID.

        Returns:
            The number of common prefix blocks for all requests with allocated
            KV cache.
        """

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the longest cache hit prefix of the blocks that is not longer than
        `max_length`. The prefix should be a common prefix hit for all the
        kv cache groups in `kv_cache_group_ids`. If no cache hit is found,
        return an empty list.
        If eagle is enabled, drop the last matched block to force recompute the
        last block to get the required hidden states for eagle drafting head.
        Need to be customized for each attention type.

        Args:
            block_hashes: The block hashes of the request.
            max_length: The maximum length of the cache hit prefix.
            kv_cache_group_ids: The ids of the kv cache groups.
            block_pool: The block pool.
            kv_cache_spec: The kv cache spec.
            use_eagle: Whether to use eagle.

        Returns:
            A list of cached blocks with skipped blocks replaced by null block
            for each kv cache group in `kv_cache_group_ids`.
            Return a list of length `len(kv_cache_group_ids)`, where the i-th
            element is a list of cached blocks for the i-th kv cache group
            in `kv_cache_group_ids`.
            For example, sliding window manager should return a list like
            ([NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)]) for block size 4
            and sliding window 8 and len(kv_cache_group_ids) = 1.
        """

        raise NotImplementedError

    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        """
        Remove and free the blocks that are no longer needed for attention computation.
        The removed blocks should be replaced by null_block.

        This function depends on `get_num_skipped_tokens`, which need to be implemented
        differently for each attention type.

        Args:
            request_id: The request ID.
            num_computed_tokens: The number of tokens that have been computed.
        """
        # Remove the blocks that will be skipped during attention computation.
        num_skipped_tokens = self.get_num_skipped_tokens(num_computed_tokens)
        if num_skipped_tokens <= 0:
            # This indicates that ALL tokens are inside attention window.
            # Thus we do not need to free any blocks outside attention window.
            # A typical case is full attention that we never free any token
            # before the request is finished.
            return
        num_skipped_blocks = num_skipped_tokens // self.block_size
        blocks = self.req_to_blocks[request_id]
        removed_blocks: list[KVCacheBlock] = []
        # Because the block starts from index 0, the num_skipped_block-th block
        # corresponds to index num_skipped_blocks - 1.
        for i in range(num_skipped_blocks - 1, -1, -1):
            if blocks[i] == self._null_block:
                # If the block is already a null block, the blocks before it
                # should also have been set to null blocks by the previous calls
                # to this function.
                break
            removed_blocks.append(blocks[i])
            blocks[i] = self._null_block
        self.block_pool.free_blocks(removed_blocks)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        # The default behavior is to not skip any tokens.
        return 0


class FullAttentionManager(SingleTypeKVCacheManager):
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(
            kv_cache_spec, (FullAttentionSpec, ChunkedLocalAttentionSpec)
        ), (
            "FullAttentionManager can only be used for full attention "
            "and chunked local attention groups"
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        if dcp_world_size * pcp_world_size > 1:
            block_size *= dcp_world_size * pcp_world_size
        max_num_blocks = max_length // block_size
        for block_hash in itertools.islice(block_hashes, max_num_blocks):
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        if self.head_grouped:
            # Head-grouped disables prefix caching, so every block has
            # ref_cnt == 1 and the cascade-attention condition
            # (ref_cnt == num_running) is false from the first block.
            return 0
        blocks = self.get_req_blocks(running_request_id)
        num_running = len(self.req_to_blocks)
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == num_running:
                num_common_blocks += 1
            else:
                break
        return num_common_blocks


class SlidingWindowManager(SingleTypeKVCacheManager):
    def __init__(
        self, kv_cache_spec: SlidingWindowSpec, block_pool: BlockPool, **kwargs
    ) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window
        self._null_block = block_pool.null_block

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, SlidingWindowSpec), (
            "SlidingWindowManager can only be used for sliding window groups"
        )
        assert dcp_world_size == 1, "DCP not support sliding window attn now."
        assert pcp_world_size == 1, "PCP not support sliding window attn now."

        # The number of contiguous blocks needed for prefix cache hit.
        # -1 since the input token itself is also included in the window
        sliding_window_contiguous_blocks = cdiv(
            kv_cache_spec.sliding_window - 1, kv_cache_spec.block_size
        )
        if use_eagle:
            # Need to drop the last matched block if eagle is enabled. For
            # sliding window layer, we achieve this by increasing the number of
            # contiguous blocks needed for prefix cache hit by one and dropping
            # the last matched block.
            sliding_window_contiguous_blocks += 1

        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
        # optimize the time complexity from O(max_num_blocks) to
        # O(max_num_blocks / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks),
        # which is good for low cache hit rate scenarios.
        max_num_blocks = max_length // kv_cache_spec.block_size
        computed_blocks = tuple(
            [block_pool.null_block] * max_num_blocks
            for _ in range(len(kv_cache_group_ids))
        )
        num_contiguous_blocks = 0
        match_found = False
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed[i] = cached
                num_contiguous_blocks += 1
                if num_contiguous_blocks >= sliding_window_contiguous_blocks:
                    # Trim the trailing blocks.
                    # E.g., [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                    # when sliding_window_contiguous_blocks=2.
                    for computed in computed_blocks:
                        del computed[i + num_contiguous_blocks :]
                    match_found = True
                    break
            else:
                num_contiguous_blocks = 0
        if not match_found:
            # The first `num_contiguous_blocks` is a cache hit even if
            # `num_contiguous_blocks < sliding_window_contiguous_blocks`.
            for computed in computed_blocks:
                del computed[num_contiguous_blocks:]
        if use_eagle and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        For sliding window, this corresponds to the tokens that are prior to
        the current sliding window.

        Example:
        sliding_window=4, num_computed_tokens=7

        Tokens:   [ 0  1  2  3  4  5  6  7 ]
                  | ---- computed -----|
                                         ^ next token to be computed
                               |-----------| sliding window for next token
                  |--skipped---|

        The current window contains tokens 4~7. Tokens 0~3 will be skipped for
        attention computation since they are outside the sliding window.
        Thus, get_num_skipped_tokens(7) == 4.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        return num_computed_tokens - self.sliding_window + 1

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        NOTE(Chen): The prefix blocks are null blocks for sliding window layers.
        So it's not correct to count ref_cnt like FullAttentionManager. Return
        0 here for correctness. Need to support cascade attention + sliding
        window in the future.
        """
        return 0


class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):
    def __init__(
        self, kv_cache_spec: ChunkedLocalAttentionSpec, block_pool: BlockPool, **kwargs
    ) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size
        self._null_block = block_pool.null_block

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        For chunked local attention, we need to find the longest cache hit
        prefix of the blocks that is not longer than `max_length`. The prefix
        should be a common prefix hit for all the kv cache groups in
        `kv_cache_group_ids`. If no cache hit is found, return an empty list.
        note we mark as computed if the whole block is outside of the local
        window, and set the block as null. Examples:

        1. Attention chunk size of 8, block size of 4, max length of 15
        for next token at 15th (zero-indexed), 8th - 14th tokens are in
        the window(needs lookup), 0th - 7th are not in the window,
        so they are already marked as computed. We check the complete
        block3 (8th - 11th tokens), Assume block 3 is hit, we will return
        [null, null, block 3], otherwise, we return [null, null]

        2. Attention chunk size of 8, block size of 4, max length of 16
        for next token at 16th (zero-indexed), 0th - 15th tokens are not
        in the window, so they are already marked as computed.
        we return 4 blocks[null, null, null, null]

        Args:
            block_hashes: The block hashes of the request.
            max_length: The maximum length of the cache hit prefix.
            kv_cache_group_ids: The ids of the kv cache groups.
            block_pool: The block pool.
            kv_cache_spec: The kv cache spec.
            use_eagle: Whether to use eagle.

        Returns:
            A list of cached blocks
        """
        assert isinstance(kv_cache_spec, ChunkedLocalAttentionSpec), (
            "ChunkedLocalAttentionManager can only be used for "
            + "chunked local attention groups"
        )
        assert use_eagle is False, (
            "Hybrid KV cache is not supported for " + "eagle + chunked local attention."
        )
        assert dcp_world_size == 1, "DCP not support chunked local attn now."
        assert pcp_world_size == 1, "PCP not support chunked local attn now."
        max_num_blocks = max_length // kv_cache_spec.block_size
        if max_length > 0:
            local_attention_start_idx = (
                max_length
                // kv_cache_spec.attention_chunk_size
                * kv_cache_spec.attention_chunk_size
            )
        else:
            local_attention_start_idx = 0
        # we marked blocks out of window as computed
        # with null blocks, and blocks inside window based on cache lookup
        # result [null] [null] ... [null] [hit block 1 (1st block contain
        # last window)] [hit block 2] ... [hit block x]
        local_attention_start_block_idx = (
            local_attention_start_idx // kv_cache_spec.block_size
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [block_pool.null_block] * local_attention_start_block_idx
            for _ in range(len(kv_cache_group_ids))
        )
        for i in range(local_attention_start_block_idx, max_num_blocks):
            block_hash = block_hashes[i]
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        return computed_blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        For chunked local attention, this corresponds to the tokens that are on
        the left side of the current chunk.

        Example 1:
        chunk size = 8, num_computed_tokens = 13
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | ----- computed ---------------|
                                                  ^^ next token to be computed
                                   |----------------| <-- attention window for
                                                          next token
                 |--- skipped -----|
        Output: get_num_skipped_tokens(13) == 8

        Example 2:
        chunk size = 8, num_computed_tokens = 8
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | --- computed ---|
                                     ^ next token to be computed
                                   |--| <-- attention window for next token
                 | --- skipped ----|
        Output: get_num_skipped_tokens(8) == 8

        Example 3:
        chunk size = 8, num_computed_tokens = 7
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 |---computed---|
                                 ^ next token to be computed
                 |-----------------| <-- attention window for next token
                 no token should be skipped.
        Output: get_num_skipped_tokens(7) == 0

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        num_skipped_tokens = (
            num_computed_tokens // self.attention_chunk_size
        ) * self.attention_chunk_size
        return num_skipped_tokens

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        cascade attention is not supported by chunked local attention.
        """
        return 0


class MambaManager(SingleTypeKVCacheManager):
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, MambaSpec), (
            "MambaManager can only be used for mamba groups"
        )
        assert dcp_world_size == 1, "DCP not support mamba now."
        assert pcp_world_size == 1, "PCP not support mamba now."
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )

        max_num_blocks = max_length // kv_cache_spec.block_size
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    # the hit length logic later assumes:
                    #  hit_length = len(hit_blocks_other_attn[0])
                    #               * self.other_block_size
                    # so we insert dummy blocks at the beginning:
                    computed.extend([block_pool.null_block] * i)
                    computed.append(cached)
                break  # we just need the last match - early stopping

        return computed_blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        cascade attention is not supported by mamba
        """
        return 0

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
    ) -> int:
        # Allocate extra `num_speculative_blocks` blocks for
        # speculative decoding (MTP/EAGLE) with linear attention.
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.kv_cache_spec.num_speculative_blocks > 0:
            num_tokens += (
                self.kv_cache_spec.block_size
                * self.kv_cache_spec.num_speculative_blocks
            )
        return super().get_num_blocks_to_allocate(
            request_id, num_tokens, new_computed_blocks
        )

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int
    ) -> list[KVCacheBlock]:
        # Allocate extra `num_speculative_blocks` blocks for
        # speculative decoding (MTP/EAGLE) with linear attention.
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.kv_cache_spec.num_speculative_blocks > 0:
            num_tokens += (
                self.kv_cache_spec.block_size
                * self.kv_cache_spec.num_speculative_blocks
            )
        return super().allocate_new_blocks(request_id, num_tokens)


class CrossAttentionManager(SingleTypeKVCacheManager):
    """Manager for cross-attention KV cache in encoder-decoder models."""

    def save_new_computed_blocks(
        self, request_id: str, new_computed_blocks: Sequence[KVCacheBlock]
    ) -> None:
        # We do not cache blocks for cross-attention to be shared between
        # requests, so  `new_computed_blocks` should always be empty.
        assert len(new_computed_blocks) == 0

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        # We do not cache blocks for cross-attention to be shared between
        # requests, so this method is not relevant.
        raise ValueError("Should not be called as prefix caching is disabled.")

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        # Cross-attention blocks contain request-specific encoder states
        # and are not shared between different requests
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, CrossAttentionSpec), (
            "CrossAttentionManager can only be used for cross-attention groups"
        )
        # Cross-attention does not benefit from prefix caching since:
        # 1. Encoder states are unique per request (different audio/image
        #    inputs)
        # 2. Encoder states are computed once per request, not incrementally
        # 3. No reusable prefix exists between different multimodal inputs
        # Return empty blocks to indicate no cache hits
        raise NotImplementedError("CrossAttentionManager does not support caching")


spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    MLAAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    MambaSpec: MambaManager,
    CrossAttentionSpec: CrossAttentionManager,
    # Head-group paging reuses FullAttentionManager — prefix-caching
    # paths are unreachable (disabled at the config layer) and the
    # remaining alloc/free paths are spec-agnostic.
    HeadGroupedAttentionSpec: FullAttentionManager,
}


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec, **kwargs
) -> SingleTypeKVCacheManager:
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
