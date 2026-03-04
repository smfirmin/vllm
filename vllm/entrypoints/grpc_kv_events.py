# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV event bridge for gRPC SubscribeKvEvents.

Bridges vLLM's internal ZMQ KV event publisher to gRPC server-streaming.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import grpc
import msgspec
import zmq
import zmq.asyncio

from vllm.config.kv_events import KVEventsConfig
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    ZmqEventPublisher,
)
from vllm.distributed.kv_events import (
    KVEventBatch as InternalKVEventBatch,
)
from vllm.grpc import vllm_engine_pb2
from vllm.logger import init_logger

logger = init_logger(__name__)

_SEQ_BYTES = 8
_POLL_TIMEOUT_MS = 200
_MASK_64 = (1 << 64) - 1
_INT64_SIGN_BIT = 1 << 63
_INT64_MOD = 1 << 64
_EVENT_ID_SEQ_MASK = (1 << 32) - 1
_REPLAY_IDLE_TIMEOUT_SECONDS = 5.0
_ZMQ_SUB = zmq.SUB  # type: ignore[attr-defined]
_ZMQ_DEALER = zmq.DEALER  # type: ignore[attr-defined]
_ZMQ_SUBSCRIBE = zmq.SUBSCRIBE  # type: ignore[attr-defined]
_ZMQ_ERROR = zmq.ZMQError  # type: ignore[attr-defined]
_CACHE_LEVEL_BY_MEDIUM = {
    "GPU": 0,
    "CPU": 1,
}


@dataclass(frozen=True)
class KVEventStreamConfig:
    endpoint: str
    replay_endpoint: str | None
    topic: str


class GrpcKVEventBridge:
    """Expose vLLM KV events as gRPC KvEventBatch messages."""

    _decoder = msgspec.msgpack.Decoder(type=InternalKVEventBatch)

    def __init__(
        self,
        config: KVEventsConfig | None,
        data_parallel_rank: int = 0,
    ):
        if (
            config is None
            or not config.enable_kv_cache_events
            or config.publisher != "zmq"
        ):
            self._config = None
            return

        endpoint = ZmqEventPublisher.offset_endpoint_port(
            config.endpoint, data_parallel_rank
        )
        replay_endpoint = ZmqEventPublisher.offset_endpoint_port(
            config.replay_endpoint, data_parallel_rank
        )

        normalized_endpoint = self._normalize_endpoint_for_connect(endpoint)
        if normalized_endpoint is None:
            self._config = None
            return

        self._config = KVEventStreamConfig(
            endpoint=normalized_endpoint,
            replay_endpoint=self._normalize_endpoint_for_connect(replay_endpoint),
            topic=config.topic,
        )

    @property
    def enabled(self) -> bool:
        return self._config is not None

    async def stream(
        self,
        start_sequence_number: int,
        context: grpc.aio.ServicerContext,
    ) -> AsyncGenerator[vllm_engine_pb2.KvEventBatch, None]:
        """Yield KV batches from replay (optional) and then live stream."""
        if self._config is None:
            return

        sub_socket: zmq.asyncio.Socket | None = None
        replay_socket: zmq.asyncio.Socket | None = None

        try:
            zmq_ctx = zmq.asyncio.Context.instance()
            sub_socket = zmq_ctx.socket(_ZMQ_SUB)
            sub_socket.connect(self._config.endpoint)
            sub_socket.setsockopt_string(_ZMQ_SUBSCRIBE, self._config.topic)

            last_seq = -1

            # Replay first, if requested and endpoint is configured.
            if start_sequence_number > 0 and self._config.replay_endpoint:
                # Use DEALER for replay because server-side ROUTER can emit
                # multiple replay responses for one request.
                replay_socket = zmq_ctx.socket(_ZMQ_DEALER)
                try:
                    replay_socket.connect(self._config.replay_endpoint)

                    replay_req = int(start_sequence_number).to_bytes(
                        _SEQ_BYTES, byteorder="big", signed=False
                    )
                    # ROUTER replay service expects [identity, empty, start_seq].
                    await replay_socket.send_multipart((b"", replay_req))
                    replay_idle_deadline = (
                        time.monotonic() + _REPLAY_IDLE_TIMEOUT_SECONDS
                    )

                    while True:
                        if self._context_cancelled(context):
                            return

                        if not await replay_socket.poll(_POLL_TIMEOUT_MS):
                            if time.monotonic() >= replay_idle_deadline:
                                logger.warning(
                                    "Replay endpoint idle timeout after %.2fs "
                                    "(start_sequence_number=%s, endpoint=%s). "
                                    "Falling back to live stream.",
                                    _REPLAY_IDLE_TIMEOUT_SECONDS,
                                    start_sequence_number,
                                    self._config.replay_endpoint,
                                )
                                break
                            continue

                        replay_idle_deadline = (
                            time.monotonic() + _REPLAY_IDLE_TIMEOUT_SECONDS
                        )
                        frames = await replay_socket.recv_multipart()
                        if len(frames) == 2:
                            seq_bytes, payload = frames
                        elif len(frames) == 3 and frames[0] == b"":
                            _, seq_bytes, payload = frames
                        else:
                            logger.warning(
                                "Invalid replay frame length: %s (expected 2 or 3)",
                                len(frames),
                            )
                            continue

                        if len(seq_bytes) != _SEQ_BYTES:
                            logger.warning(
                                "Invalid replay sequence length: %s (expected %s)",
                                len(seq_bytes),
                                _SEQ_BYTES,
                            )
                            continue

                        if seq_bytes == ZmqEventPublisher.END_SEQ and not payload:
                            break
                        replay_seq = int.from_bytes(
                            seq_bytes, byteorder="big", signed=False
                        )

                        if replay_seq < start_sequence_number:
                            continue
                        if replay_seq <= last_seq:
                            continue

                        batch = self._decode_and_convert_batch(replay_seq, payload)
                        if batch is None:
                            continue
                        last_seq = replay_seq
                        yield batch
                except _ZMQ_ERROR as exc:
                    logger.warning(
                        "Replay unavailable (start_sequence_number=%s, endpoint=%s): "
                        "%s. Falling back to live stream.",
                        start_sequence_number,
                        self._config.replay_endpoint,
                        exc,
                    )

            # Live stream.
            while True:
                if self._context_cancelled(context):
                    return

                if not await sub_socket.poll(_POLL_TIMEOUT_MS):
                    continue

                frames = await sub_socket.recv_multipart()
                if len(frames) != 3:
                    logger.warning(
                        "Invalid SUB frame length: %s (expected 3)", len(frames)
                    )
                    continue

                _, seq_bytes, payload = frames
                if len(seq_bytes) != _SEQ_BYTES:
                    logger.warning(
                        "Invalid SUB sequence length: %s (expected %s)",
                        len(seq_bytes),
                        _SEQ_BYTES,
                    )
                    continue

                seq = int.from_bytes(seq_bytes, byteorder="big", signed=False)
                if seq < start_sequence_number:
                    continue
                if seq <= last_seq:
                    continue

                batch = self._decode_and_convert_batch(seq, payload)
                if batch is None:
                    continue
                last_seq = seq
                yield batch
        finally:
            if replay_socket is not None:
                replay_socket.close(linger=0)
            if sub_socket is not None:
                sub_socket.close(linger=0)

    @classmethod
    def _decode_and_convert_batch(
        cls, sequence_number: int, payload: bytes
    ) -> vllm_engine_pb2.KvEventBatch | None:
        try:
            batch = cls._decoder.decode(payload)
        except Exception as exc:
            logger.warning(
                "Failed to decode KV event batch seq=%s: %s", sequence_number, exc
            )
            return None
        return cls._convert_batch(sequence_number, batch)

    @classmethod
    def _convert_batch(
        cls, sequence_number: int, batch: InternalKVEventBatch
    ) -> vllm_engine_pb2.KvEventBatch:
        out_events: list[vllm_engine_pb2.KvCacheEvent] = []

        for idx, event in enumerate(batch.events):
            # Keep event_id within uint64 bounds even for very long-lived streams.
            event_id = ((sequence_number & _EVENT_ID_SEQ_MASK) << 32) | idx
            if isinstance(event, BlockStored):
                out_events.append(cls._convert_block_stored(event, event_id))
            elif isinstance(event, BlockRemoved):
                out_events.append(cls._convert_block_removed(event, event_id))
            elif isinstance(event, AllBlocksCleared):
                out_events.append(
                    vllm_engine_pb2.KvCacheEvent(
                        event_id=event_id, cleared=vllm_engine_pb2.KvCacheCleared()
                    )
                )

        out_batch = vllm_engine_pb2.KvEventBatch(
            sequence_number=sequence_number,
            timestamp=batch.ts,
            events=out_events,
        )
        if batch.data_parallel_rank is not None:
            out_batch.dp_rank = batch.data_parallel_rank
        return out_batch

    @classmethod
    def _convert_block_stored(
        cls, event: BlockStored, event_id: int
    ) -> vllm_engine_pb2.KvCacheEvent:
        tokens = list(event.token_ids)
        block_size = max(int(event.block_size), 1)
        cache_level = cls._cache_level_from_medium(event.medium)

        blocks: list[vllm_engine_pb2.KvBlock] = []
        for block_idx, block_hash in enumerate(event.block_hashes):
            start = block_idx * block_size
            end = min(start + block_size, len(tokens))
            block = vllm_engine_pb2.KvBlock(
                block_hash=cls._to_signed_int64(block_hash),
                token_ids=tokens[start:end],
                block_size=event.block_size,
            )
            if event.lora_id is not None:
                block.lora_id = event.lora_id
            if cache_level is not None:
                block.cache_level = cache_level
            blocks.append(block)

        stored = vllm_engine_pb2.KvBlocksStored(blocks=blocks)
        if event.parent_block_hash is not None:
            stored.parent_block_hash = cls._to_signed_int64(event.parent_block_hash)
        return vllm_engine_pb2.KvCacheEvent(event_id=event_id, stored=stored)

    @classmethod
    def _convert_block_removed(
        cls, event: BlockRemoved, event_id: int
    ) -> vllm_engine_pb2.KvCacheEvent:
        removed = vllm_engine_pb2.KvBlocksRemoved(
            block_hashes=[cls._to_signed_int64(h) for h in event.block_hashes]
        )
        cache_level = cls._cache_level_from_medium(event.medium)
        if cache_level is not None:
            removed.cache_level = cache_level
        return vllm_engine_pb2.KvCacheEvent(event_id=event_id, removed=removed)

    @staticmethod
    def _to_signed_int64(value: int | bytes) -> int:
        if isinstance(value, bytes):
            unsigned = int.from_bytes(value, byteorder="big", signed=False) & _MASK_64
        else:
            unsigned = int(value) & _MASK_64

        if unsigned >= _INT64_SIGN_BIT:
            return unsigned - _INT64_MOD
        return unsigned

    @staticmethod
    def _cache_level_from_medium(medium: str | None) -> int | None:
        if medium is None:
            return None
        return _CACHE_LEVEL_BY_MEDIUM.get(medium.upper())

    @staticmethod
    def _context_cancelled(context: grpc.aio.ServicerContext) -> bool:
        # `done()` and `cancelled()` are available in grpc aio ServicerContext.
        # Guard with getattr for compatibility in unit tests/mocks.
        done = getattr(context, "done", None)
        if callable(done) and done():
            return True
        cancelled = getattr(context, "cancelled", None)
        return bool(callable(cancelled) and cancelled())

    @staticmethod
    def _normalize_endpoint_for_connect(endpoint: str | None) -> str | None:
        """Normalize bind-style wildcard endpoints into connectable addresses."""
        if endpoint is None:
            return None
        if endpoint.startswith("tcp://*"):
            return endpoint.replace("*", "127.0.0.1", 1)
        return endpoint
