"""
Token bucket rate limiter.

Can be used as a standalone ``await limiter.acquire()`` check or as a
decorator via ``@limiter.limit()``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from ._exceptions import RateLimitExceeded


class TokenBucketLimiter:
    """Async token bucket rate limiter.

    Args:
        capacity: Maximum number of tokens (burst size).
        refill_rate: Tokens added per second.

    Example::

        limiter = TokenBucketLimiter(capacity=10, refill_rate=2.0)

        # Manual
        if not await limiter.acquire():
            raise RateLimitExceeded()

        # As decorator
        @limiter.limit()
        async def handler(): ...
    """

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Try to consume one token. Returns ``True`` on success."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
            self._last_refill = now

            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False

    def limit(self) -> Callable[..., Any]:
        """Decorator that raises :class:`~._exceptions.RateLimitExceeded` when empty."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            async def wrapper(*args: object, **kwargs: object) -> object:
                if not await self.acquire():
                    raise RateLimitExceeded()
                return await func(*args, **kwargs)
            return wrapper
        return decorator
