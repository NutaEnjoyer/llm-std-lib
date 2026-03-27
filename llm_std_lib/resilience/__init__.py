"""
Resilience sub-package for llm_std_lib.

Provides fault-tolerance primitives for LLM provider calls:
- :class:`~.circuit_breaker.CircuitBreaker` — stops calls to failing providers.
- :class:`~.retry.RetryPolicy` — exponential backoff with jitter.
- :class:`~.engine.ResilienceEngine` — combines breaker + retry + timeout.
- :class:`~.limiter.TokenBucketLimiter` — token bucket rate limiting.
- :class:`~.fallback.FallbackChain` — ordered provider fallback.
- :class:`~.backend.InMemoryBackend` — default in-process state backend.
"""

from ._exceptions import CircuitOpenError, MaxRetriesExceeded, RateLimitExceeded
from .backend import InMemoryBackend, StateBackend
from .circuit_breaker import CircuitBreaker
from .engine import ResilienceEngine
from .limiter import TokenBucketLimiter
from .metrics import BreakerMetrics, RetryMetrics
from .retry import RetryPolicy
from .state import BreakerState

__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "InMemoryBackend",
    "MaxRetriesExceeded",
    "RateLimitExceeded",
    "ResilienceEngine",
    "RetryPolicy",
    "StateBackend",
    "TokenBucketLimiter",
    "BreakerMetrics",
    "RetryMetrics",
    "BreakerState",
]
