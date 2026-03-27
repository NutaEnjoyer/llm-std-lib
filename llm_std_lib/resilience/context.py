"""Async context manager for manual success/failure recording."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

    from .engine import ResilienceEngine


class ResilienceContext:
    """Async context manager that records the outcome of a protected block.

    Usage::

        async with engine.protect() as ctx:
            result = await call_provider()
    """

    def __init__(self, engine: ResilienceEngine) -> None:
        self.engine = engine

    async def __aenter__(self) -> ResilienceContext:
        await self.engine._breaker.before_call()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc is None:
            await self.engine._breaker.record_success()
        else:
            await self.engine._breaker.record_failure()
        return False
