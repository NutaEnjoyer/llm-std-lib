"""
Rate-limiter middleware.

Enforces a client-side token bucket rate limit on outbound LLM requests.
Built on :class:`~llm_std_lib.resilience.limiter.TokenBucketLimiter`.
"""

from __future__ import annotations

from llm_std_lib.exceptions import LLMRateLimitError
from llm_std_lib.middleware.base import BaseMiddleware
from llm_std_lib.resilience.limiter import TokenBucketLimiter
from llm_std_lib.types import RequestContext


class RateLimiterMiddleware(BaseMiddleware):
    """Limits outbound requests using a token bucket algorithm.

    Args:
        capacity: Maximum burst size (number of requests).
        refill_rate: Tokens refilled per second (sustained throughput).

    Raises:
        :class:`~llm_std_lib.exceptions.LLMRateLimitError`: When the bucket
            is empty and no token can be acquired.

    Example::

        # Allow bursts of 10 requests, sustained at 2 req/s
        stack = MiddlewareStack([RateLimiterMiddleware(capacity=10, refill_rate=2.0)])
    """

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self._limiter = TokenBucketLimiter(capacity=capacity, refill_rate=refill_rate)

    async def pre_request(self, ctx: RequestContext) -> RequestContext:
        if not await self._limiter.acquire():
            raise LLMRateLimitError(
                f"Client-side rate limit exceeded (capacity={self._limiter.capacity}, "
                f"refill_rate={self._limiter.refill_rate}/s)."
            )
        return ctx
