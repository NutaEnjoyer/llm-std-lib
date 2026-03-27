"""
ResilienceEngine — combines circuit breaker, retry, and timeout.

The engine is the main entry point for protecting calls to an LLM provider.
It can be used as a decorator, an async context manager, or directly via
:meth:`execute`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

from ._exceptions import CircuitOpenError, MaxRetriesExceeded
from .circuit_breaker import CircuitBreaker
from .metrics import BreakerMetrics, RetryMetrics
from .retry import RetryPolicy
from .state import BreakerState

T = TypeVar("T")

logger = logging.getLogger("llm_std_lib.resilience.engine")


class ResilienceEngine:
    """Combines :class:`~.circuit_breaker.CircuitBreaker` and :class:`~.retry.RetryPolicy`.

    Args:
        breaker: The circuit breaker instance to use.
        retryer: The retry policy to apply on transient failures.
        timeout: Optional per-call timeout in seconds.

    Example::

        backend = InMemoryBackend()
        breaker = CircuitBreaker(backend, key="openai")
        policy  = RetryPolicy(max_attempts=3, base_delay=0.5)
        engine  = ResilienceEngine(breaker, retryer=policy, timeout=10.0)

        result = await engine.execute(lambda: provider.complete(ctx))
    """

    def __init__(
        self,
        breaker: CircuitBreaker,
        retryer: RetryPolicy,
        timeout: float | None = None,
    ) -> None:
        self._breaker = breaker
        self._retryer = retryer
        self.timeout = timeout
        self._retry_metrics = RetryMetrics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def protect(self, func: Callable[..., Any] | None = None) -> Any:
        """Use as a decorator or call with no arguments for a context manager.

        As a decorator::

            @engine.protect
            async def handler(): ...

        As a context manager::

            async with engine.protect() as ctx:
                result = await call_service()
        """
        if func is None:
            from .context import ResilienceContext
            return ResilienceContext(self)

        @wraps(func)
        async def wrapper(*args: object, **kwargs: object) -> object:
            return await self.execute(lambda: func(*args, **kwargs))

        return wrapper

    async def execute(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute *func* with circuit breaker + retry protection.

        Args:
            func: A zero-argument async callable to protect.

        Returns:
            The return value of *func*.

        Raises:
            CircuitOpenError: If the circuit is OPEN and recovery timeout
                has not elapsed.
            MaxRetriesExceeded: If all retry attempts are exhausted.
            asyncio.TimeoutError: If *timeout* is set and the call exceeds it.
        """
        last_error: Exception | None = None

        for attempt in range(self._retryer.max_attempts + 1):
            await self._breaker.before_call()

            try:
                if self.timeout:
                    result = await asyncio.wait_for(func(), self.timeout)
                else:
                    result = await func()

                await self._breaker.record_success()
                return result

            except asyncio.TimeoutError:
                logger.warning("Call timed out after %.1fs", self.timeout)
                raise

            except Exception as exc:
                last_error = exc
                if isinstance(exc, CircuitOpenError):
                    raise

                await self._breaker.record_failure()

                if not self._retryer.is_retryable(exc):
                    raise

                if attempt < self._retryer.max_attempts:
                    self._retry_metrics.total_retries += 1
                    delay = self._retryer._compute_delay(attempt)
                    logger.debug(
                        "Retry attempt %d/%d after %.2fs due to: %s",
                        attempt + 1,
                        self._retryer.max_attempts,
                        delay,
                        type(exc).__name__,
                    )
                    await self._retryer.wait(attempt)

        self._retry_metrics.total_retry_exhausted += 1
        logger.warning(
            "Max retries (%d) exhausted. Last error: %s",
            self._retryer.max_attempts,
            type(last_error).__name__,
        )
        raise MaxRetriesExceeded() from last_error

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    async def current_state(self) -> BreakerState:
        return await self._breaker.current_state()

    async def failure_ratio(self) -> float:
        return await self._breaker.failure_ratio()

    async def is_open(self) -> bool:
        return await self._breaker.is_open()

    def breaker_metrics(self) -> BreakerMetrics:
        return self._breaker.get_metrics()

    def retry_metrics(self) -> RetryMetrics:
        return self._retry_metrics
