"""
State backends for the circuit breaker.

StateBackend is an abstract interface; InMemoryBackend is the default
implementation. Future versions may add a RedisBackend for distributed
circuit breaking across multiple processes.
"""

from __future__ import annotations

import asyncio

from .state import BreakerState


class StateBackend:
    """Abstract state backend for circuit breaker persistence."""

    async def get(self, key: str) -> BreakerState | None:
        raise NotImplementedError

    async def compare_and_set(
        self,
        key: str,
        expected: BreakerState | None,
        new: BreakerState,
    ) -> bool:
        raise NotImplementedError


class InMemoryBackend(StateBackend):
    """Thread-safe in-process backend using asyncio locks and optimistic CAS.

    Suitable for single-process deployments. For multi-process/distributed
    setups a Redis-backed backend should be used instead.
    """

    def __init__(self) -> None:
        self._data: dict[str, BreakerState] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get(self, key: str) -> BreakerState | None:
        self._locks.setdefault(key, asyncio.Lock())
        async with self._locks[key]:
            return self._data.get(key)

    async def compare_and_set(
        self,
        key: str,
        expected: BreakerState | None,
        new: BreakerState,
    ) -> bool:
        self._locks.setdefault(key, asyncio.Lock())
        async with self._locks[key]:
            current = self._data.get(key)

            if current is None:
                if expected is None:
                    self._data[key] = new
                    return True
                return False

            if current.version != expected.version:  # type: ignore[union-attr]
                return False

            self._data[key] = new
            return True
