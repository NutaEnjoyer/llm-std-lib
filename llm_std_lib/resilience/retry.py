"""
Retry policy with exponential backoff and jitter.

Configurable via constructor; the ``retryable`` argument accepts either a
tuple of exception types or a callable for full flexibility.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable


class RetryPolicy:
    """Exponential backoff retry policy with optional full jitter.

    Args:
        max_attempts: Maximum number of retry attempts (not counting the
            initial call). Default: ``3``.
        base_delay: Initial delay in seconds. Default: ``0.5``.
        max_delay: Upper bound on computed delay before jitter. Default: ``30.0``.
        exponential: If ``True``, delay grows as ``base_delay * 2^attempt``.
        jitter: If ``True``, applies full jitter: ``uniform(0, delay)``.
        retryable: Exception types to retry, or a callable
            ``(exc) -> bool``. Default: all exceptions.

    Example::

        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.5,
            retryable=(LLMRateLimitError, LLMTimeoutError),
        )
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        exponential: bool = True,
        jitter: bool = True,
        retryable: tuple[type[Exception], ...] | Callable[[Exception], bool] = (
            Exception,
        ),
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential = exponential
        self.jitter = jitter
        self.retryable = retryable

    def is_retryable(self, exc: Exception) -> bool:
        """Return ``True`` if *exc* should trigger a retry."""
        if isinstance(self.retryable, tuple):
            return isinstance(exc, self.retryable)
        return self.retryable(exc)

    def _compute_delay(self, attempt: int) -> float:
        delay = self.base_delay * (2**attempt) if self.exponential else self.base_delay
        delay = min(delay * 0.5, self.max_delay)
        if self.jitter:
            delay = random.uniform(0, delay)
        return delay

    async def wait(self, attempt: int) -> None:
        """Sleep for the computed backoff duration."""
        await asyncio.sleep(self._compute_delay(attempt))
