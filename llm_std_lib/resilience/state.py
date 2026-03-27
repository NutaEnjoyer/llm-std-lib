"""
Circuit breaker state and transition helpers.

BreakerState is immutable (frozen dataclass). All mutations return a new
instance via ``evolve()``, enabling optimistic concurrency in the backend.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from ._types import BreakerStatus, WindowType


@dataclass(frozen=True)
class BreakerState:
    status: BreakerStatus
    failure_count: int
    success_count: int
    half_open_max_calls: int
    half_open_in_flight: int
    half_open_required_successes: int
    opened_at: float | None
    version: int

    history: deque  # type: ignore[type-arg]
    window_type: WindowType
    window_size: int
    window_seconds: int

    @staticmethod
    def initial() -> BreakerState:
        return BreakerState(
            status=BreakerStatus.CLOSED,
            failure_count=0,
            success_count=0,
            half_open_max_calls=5,
            half_open_in_flight=0,
            half_open_required_successes=3,
            opened_at=None,
            version=0,
            history=deque(maxlen=20),
            window_type=WindowType.COUNT,
            window_size=20,
            window_seconds=10,
        )


def evolve(state: BreakerState, **changes: object) -> BreakerState:
    """Return a new BreakerState with the given fields replaced."""
    return BreakerState(
        status=changes.get("status", state.status),  # type: ignore[arg-type]
        failure_count=changes.get("failure_count", state.failure_count),  # type: ignore[arg-type]
        success_count=changes.get("success_count", state.success_count),  # type: ignore[arg-type]
        half_open_max_calls=changes.get("half_open_max_calls", state.half_open_max_calls),  # type: ignore[arg-type]
        half_open_in_flight=changes.get("half_open_in_flight", state.half_open_in_flight),  # type: ignore[arg-type]
        half_open_required_successes=changes.get(  # type: ignore[arg-type]
            "half_open_required_successes", state.half_open_required_successes
        ),
        opened_at=changes.get("opened_at", state.opened_at),  # type: ignore[arg-type]
        history=changes.get("history", state.history),  # type: ignore[arg-type]
        window_type=changes.get("window_type", state.window_type),  # type: ignore[arg-type]
        window_size=changes.get("window_size", state.window_size),  # type: ignore[arg-type]
        window_seconds=changes.get("window_seconds", state.window_seconds),  # type: ignore[arg-type]
        version=state.version + 1,
    )


def add_result(state: BreakerState, success: bool) -> BreakerState:
    """Append a call result to the sliding window and return a new state."""
    if state.window_type == WindowType.COUNT:
        new_history: deque = deque(state.history, maxlen=state.window_size)  # type: ignore[type-arg]
        new_history.append(success)
    else:  # TIME
        now = time.time()
        new_history = deque(list(state.history))
        new_history.append((now, success))
        while new_history and now - new_history[0][0] > state.window_seconds:
            new_history.popleft()

    return evolve(state, history=new_history)


def failure_rating(state: BreakerState) -> float:
    """Return the fraction of failures in the current window."""
    if not state.history:
        return 0.0

    if state.window_type == WindowType.COUNT:
        failures = state.history.count(False)
    else:  # TIME
        failures = sum(1 for _t, s in state.history if not s)

    return failures / len(state.history)
