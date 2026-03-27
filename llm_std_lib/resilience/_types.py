"""Enumerations for circuit breaker state and window type."""

from __future__ import annotations

from enum import Enum


class BreakerStatus(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class WindowType(str, Enum):
    COUNT = "COUNT"
    TIME = "TIME"
