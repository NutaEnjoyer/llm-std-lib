"""
Internal resilience exceptions.

These are low-level exceptions used within the resilience layer.
At the boundary (FallbackChain, LLMClient) they are mapped to the
public LLM exception hierarchy defined in llm_std_lib.exceptions.
"""

from __future__ import annotations


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is OPEN."""


class MaxRetriesExceeded(Exception):
    """Raised when all retry attempts are exhausted."""


class RateLimitExceeded(Exception):
    """Raised when the token bucket is empty."""
