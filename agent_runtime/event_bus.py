"""Asynchronous event bus primitives for the monitoring agent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional


@dataclass(slots=True)
class Event:
    """Generic event envelope used for inter-stage communication."""

    topic: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncEventBus:
    """Minimal pub/sub facility built on top of asyncio queues."""

    def __init__(self, max_queue_size: int = 32) -> None:
        self._queues: Dict[str, asyncio.Queue[Event]] = {}
        self._max_queue_size = max_queue_size
        self._lock = asyncio.Lock()
        self._closed = asyncio.Event()

    async def publish(self, event: Event) -> None:
        if self._closed.is_set():
            return
        async with self._lock:
            queue = self._queues.get(event.topic)
            if queue is None:
                queue = asyncio.Queue(self._max_queue_size)
                self._queues[event.topic] = queue
        await queue.put(event)

    async def subscribe(self, topic: str) -> AsyncIterator[Event]:
        async with self._lock:
            queue = self._queues.get(topic)
            if queue is None:
                queue = asyncio.Queue(self._max_queue_size)
                self._queues[topic] = queue
        while not self._closed.is_set():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            yield event

    async def get_queue(self, topic: str) -> asyncio.Queue[Event]:
        async with self._lock:
            queue = self._queues.get(topic)
            if queue is None:
                queue = asyncio.Queue(self._max_queue_size)
                self._queues[topic] = queue
            return queue

    async def close(self) -> None:
        self._closed.set()
        async with self._lock:
            for queue in self._queues.values():
                while not queue.empty():
                    queue.get_nowait()
            self._queues.clear()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()
