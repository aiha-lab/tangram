# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        enable_kv_cache_events: Whether to enable kv cache events.
        ragged: Whether the pool is feeding ragged paging.
            Informational only — every block is byte-equal and head-group
            agnostic; this flag switches storage to a numpy ring buffer.
        num_head_groups: Total head-groups across all layers
            (= ``num_head_groups_per_layer * num_layers``); required when
            ``ragged`` is True.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        enable_kv_cache_events: bool = False,
        ragged: bool = False,
        num_head_groups: int | None = None,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        if ragged:
            assert num_head_groups is not None and num_head_groups > 0, (
                "ragged=True requires a positive num_head_groups."
            )
        else:
            assert num_head_groups is None, (
                "num_head_groups must be None when ragged=False."
            )
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.ragged = ragged
        self.num_head_groups = num_head_groups
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Doubly linked list of free blocks (eviction candidates when
        # caching is on). Ragged paging swaps this for an int32 ring
        # buffer (see ``free_block_ids``) so alloc/free can run as bulk
        # numpy ops; prefix caching is disabled there, removing the only
        # consumer of mid-list removal.
        if ragged:
            # Block 0 is reserved as the null block; ring holds ids 1..N-1.
            self.free_block_queue: FreeKVCacheBlockQueue | None = None
            self.free_block_ids: np.ndarray = np.arange(
                1, num_gpu_blocks, dtype=np.int32
            )
            self._ring_capacity: int = num_gpu_blocks - 1
            self._ring_head: int = 0
            # Position one past the last live entry; wraps to 0 on append.
            self._ring_tail: int = self._ring_capacity
            self.num_free_blocks: int = self._ring_capacity
            # Mirror of per-block ref_cnt for hot-path bulk ops; the
            # per-block ``KVCacheBlock.ref_cnt`` is left untouched and
            # never read in this mode.
            self.ref_cnts: np.ndarray = np.zeros(num_gpu_blocks, dtype=np.int8)
            self.null_block = self.blocks[0]
            self.null_block.is_null = True
        else:
            self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
            # To represent a placeholder block with block_id=0.
            # The ref_cnt of null_block is not maintained, needs special care
            # to avoid freeing it.
            self.null_block = self.free_block_queue.popleft()
            self.null_block.is_null = True

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        new_block_hashes = request.block_hashes[num_cached_blocks:]

        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block = blocks[num_cached_blocks - 1]
                assert parent_block.block_hash is not None
                parent_block_hash = maybe_convert_block_hash(
                    get_block_hash(parent_block.block_hash)
                )

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[
                        num_cached_blocks * block_size : num_full_blocks * block_size
                    ],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                )
            )

    def _ring_pop_ids(self, num_blocks: int) -> np.ndarray:
        """Pop ``num_blocks`` ids from the ragged ring buffer.

        Caller must ensure ``num_free_blocks >= num_blocks``.
        """
        head = self._ring_head
        capacity = self._ring_capacity
        if head + num_blocks <= capacity:
            ids = self.free_block_ids[head : head + num_blocks].copy()
            new_head = head + num_blocks
        else:
            first = capacity - head
            ids = np.empty(num_blocks, dtype=np.int32)
            ids[:first] = self.free_block_ids[head:]
            ids[first:] = self.free_block_ids[: num_blocks - first]
            new_head = num_blocks - first
        self._ring_head = new_head % capacity
        self.num_free_blocks -= num_blocks
        return ids

    def _ring_append_ids(self, ids: np.ndarray) -> None:
        """Push ``ids`` into the ring buffer (FIFO; order is irrelevant
        since ragged mode has no LRU invariant)."""
        num = ids.size
        if num == 0:
            return
        tail = self._ring_tail
        capacity = self._ring_capacity
        if tail + num <= capacity:
            self.free_block_ids[tail : tail + num] = ids
            new_tail = tail + num
        else:
            first = capacity - tail
            self.free_block_ids[tail:] = ids[:first]
            self.free_block_ids[: num - first] = ids[first:]
            new_tail = num - first
        self._ring_tail = new_tail % capacity
        self.num_free_blocks += num

    def get_new_block_ids(self, num_blocks: int) -> np.ndarray:
        """Ragged fast path: int32 ndarray of newly-allocated ids.

        Bulk-updates ``ref_cnts`` in one numpy op and skips
        ``KVCacheBlock`` materialisation entirely.
        """
        assert self.ragged
        if num_blocks > self.num_free_blocks:
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the pool"
            )
        if num_blocks == 0:
            return np.empty(0, dtype=np.int32)
        ids = self._ring_pop_ids(num_blocks)
        self.ref_cnts[ids] = 1
        return ids

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if self.ragged:
            # Materialise from the ring buffer for callers still expecting
            # ``list[KVCacheBlock]``; the ragged manager uses
            # ``get_new_block_ids`` directly and skips this list build.
            ids = self.get_new_block_ids(num_blocks)
            if ids.size == 0:
                return []
            pool_blocks = self.blocks
            return [pool_blocks[int(bid)] for bid in ids.tolist()]

        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # block not found in cached_block_hash_to_block,
            # eviction is not needed
            return False

        block.reset_hash()

        if self.enable_kv_cache_events:
            # FIXME (Chen): Not sure whether we should return `hash_value`
            # or `(hash_value, group_id)` here. But it's fine now because
            # we disable hybrid kv cache manager when kv cache event is
            # enabled, so there is only one group.
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                )
            )
        return True

    def touch(self, blocks: tuple[Sequence[KVCacheBlock], ...]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        # Ragged paging force-disables prefix caching; surface any
        # accidental call rather than silently corrupting ref_cnts.
        assert not self.ragged, (
            "BlockPool.touch is not supported under ragged paging."
        )
        for blocks_per_group in blocks:
            for block in blocks_per_group:
                # ref_cnt=0 means this block is in the free list (i.e. eviction
                # candidate), so remove it.
                if block.ref_cnt == 0 and not block.is_null:
                    self.free_block_queue.remove(block)
                block.ref_cnt += 1

    def free_block_ids_array(self, ids: np.ndarray) -> None:
        """Ragged fast path: bulk-push int32 ids back into the ring
        buffer. Caller must guarantee ids are unique and exclude the null
        block (id 0); pushing the same id twice would corrupt
        ``num_free_blocks``.
        """
        assert self.ragged
        if ids.size == 0:
            return
        self.ref_cnts[ids] = 0
        self._ring_append_ids(ids)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        if self.ragged:
            # Manager bookkeeping is the authority here; only called on
            # request termination. Build an id ndarray, skip null, push.
            ids_list = [
                blk.block_id for blk in ordered_blocks if not blk.is_null
            ]
            if not ids_list:
                return
            ids = np.fromiter(ids_list, dtype=np.int32, count=len(ids_list))
            self.free_block_ids_array(ids)
            return
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
        self.free_block_queue.append_n(
            [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
        )

    def free_by_block_ids(self, block_ids: Iterable[int]) -> None:
        """Force-free physical block ids after compression resolved that no
        request still references them. Resets ``ref_cnt`` to 0 directly
        (not via ``free_blocks``) and returns them to the free pool.
        """
        if self.ragged:
            if isinstance(block_ids, np.ndarray):
                arr = block_ids.astype(np.int32, copy=False)
            else:
                arr = np.fromiter(block_ids, dtype=np.int32,
                                  count=len(block_ids))
            if arr.size == 0:
                return
            # Drop null id 0 and any id already at ref_cnt 0 (another layer
            # may have reported the same id in this step).
            live = (arr != 0) & (self.ref_cnts[arr] != 0)
            if not live.all():
                arr = arr[live]
                if arr.size == 0:
                    return
            self.free_block_ids_array(arr)
            return
        seen: set[int] = set()
        to_append: list[KVCacheBlock] = []
        for bid in block_ids:
            ibid = int(bid)
            if ibid in seen:
                continue
            seen.add(ibid)
            block = self.blocks[ibid]
            if block.is_null:
                continue
            if block.ref_cnt == 0:
                # Already freed by another layer reporting the same id.
                continue
            block.ref_cnt = 0
            to_append.append(block)
        if to_append:
            self.free_block_queue.append_n(to_append)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        if self.ragged:
            return self.num_free_blocks
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
