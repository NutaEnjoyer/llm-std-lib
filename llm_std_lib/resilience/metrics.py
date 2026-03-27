"""Lightweight metrics counters for circuit breaker and retry activity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BreakerMetrics:
    total_calls: int = 0
    total_success: int = 0
    total_failure: int = 0

    total_open_transitions: int = 0
    total_half_open_transitions: int = 0
    total_close_transitions: int = 0


@dataclass
class RetryMetrics:
    total_retries: int = 0
    total_retry_exhausted: int = 0
