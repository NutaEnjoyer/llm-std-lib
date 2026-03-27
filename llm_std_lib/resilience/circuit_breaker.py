"""
Circuit breaker for LLM provider calls.

Protects against cascading failures by stopping calls to a failing provider
once the failure ratio exceeds the configured threshold. Automatically probes
for recovery via the HALF_OPEN state after the recovery timeout elapses.

States:
    CLOSED    — normal operation, all calls pass through.
    OPEN      — calls are rejected with :class:`~._exceptions.CircuitOpenError`.
    HALF_OPEN — a limited number of probe calls are allowed through to test
                whether the provider has recovered.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from ._exceptions import CircuitOpenError
from ._types import BreakerStatus
from .backend import StateBackend
from .metrics import BreakerMetrics
from .state import BreakerState, add_result, evolve, failure_rating

logger = logging.getLogger("llm_std_lib.resilience.circuit_breaker")


class CircuitBreaker:
    """Async circuit breaker with pluggable state backend.

    Args:
        backend: State persistence backend (e.g. :class:`~.backend.InMemoryBackend`).
        key: Unique identifier for this breaker (typically the provider name).
        failure_threshold_ratio: Fraction of failures in the window that
            triggers OPEN. Default: ``0.5``.
        recovery_timeout: Seconds the breaker stays OPEN before attempting
            HALF_OPEN. Default: ``30.0``.
        on_open: Async callback invoked when the breaker transitions to OPEN.
        on_half_open: Async callback invoked on transition to HALF_OPEN.
        on_close: Async callback invoked when the breaker recovers (CLOSED).

    Example::

        backend = InMemoryBackend()
        breaker = CircuitBreaker(backend, key="openai", failure_threshold_ratio=0.5)
    """

    def __init__(
        self,
        backend: StateBackend,
        key: str,
        failure_threshold_ratio: float = 0.5,
        recovery_timeout: float = 30.0,
        on_open: Callable[[BreakerState], Awaitable[None]] | None = None,
        on_half_open: Callable[[BreakerState], Awaitable[None]] | None = None,
        on_close: Callable[[BreakerState], Awaitable[None]] | None = None,
    ) -> None:
        self.backend = backend
        self.key = key
        self.failure_threshold_ratio = failure_threshold_ratio
        self.recovery_timeout = recovery_timeout
        self.on_open = on_open
        self.on_half_open = on_half_open
        self.on_close = on_close
        self.metrics = BreakerMetrics()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_state(self) -> BreakerState:
        state = await self.backend.get(self.key)
        if state is None:
            state = BreakerState.initial()
            await self.backend.compare_and_set(self.key, None, state)
        return state

    async def _trigger_hook(self, state: BreakerState, new_state: BreakerState) -> None:
        if state.status == new_state.status:
            return

        logger.info(
            "Circuit breaker [%s]: %s -> %s",
            self.key,
            state.status.name,
            new_state.status.name,
        )

        try:
            if new_state.status == BreakerStatus.OPEN and self.on_open:
                await self.on_open(new_state)
            elif new_state.status == BreakerStatus.HALF_OPEN and self.on_half_open:
                await self.on_half_open(new_state)
            elif new_state.status == BreakerStatus.CLOSED and self.on_close:
                await self.on_close(new_state)
        except Exception:
            pass

        if new_state.status == BreakerStatus.OPEN:
            self.metrics.total_open_transitions += 1
        elif new_state.status == BreakerStatus.HALF_OPEN:
            self.metrics.total_half_open_transitions += 1
        elif new_state.status == BreakerStatus.CLOSED:
            self.metrics.total_close_transitions += 1

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    async def before_call(self) -> None:
        """Gate a call attempt — raises :class:`CircuitOpenError` if blocked."""
        while True:
            state = await self._get_state()

            if state.status == BreakerStatus.OPEN:
                elapsed = time.time() - (state.opened_at or 0)
                if elapsed < self.recovery_timeout:
                    raise CircuitOpenError()

                new_state = evolve(state, status=BreakerStatus.HALF_OPEN, half_open_in_flight=1)
                if await self.backend.compare_and_set(self.key, state, new_state):
                    await self._trigger_hook(state, new_state)
                    self.metrics.total_calls += 1
                    return
                continue

            if state.status == BreakerStatus.HALF_OPEN:
                if state.half_open_max_calls <= state.half_open_in_flight:
                    raise CircuitOpenError()
                new_state = evolve(
                    state,
                    status=BreakerStatus.HALF_OPEN,
                    half_open_in_flight=state.half_open_in_flight + 1,
                )
                if await self.backend.compare_and_set(self.key, state, new_state):
                    await self._trigger_hook(state, new_state)
                    self.metrics.total_calls += 1
                    return
                continue

            # CLOSED — always allow
            self.metrics.total_calls += 1
            return

    async def record_success(self) -> None:
        """Record a successful call and potentially close the circuit."""
        while True:
            state = await self._get_state()
            if state.status == BreakerStatus.OPEN:
                return

            new_state = add_result(state, True)
            in_flight = max(0, state.half_open_in_flight - 1)

            if (
                state.status == BreakerStatus.HALF_OPEN
                and state.success_count + 1 >= state.half_open_required_successes
            ):
                new_state = evolve(
                    new_state,
                    status=BreakerStatus.CLOSED,
                    failure_count=0,
                    success_count=0,
                    half_open_in_flight=in_flight,
                    opened_at=None,
                )
            else:
                new_state = evolve(
                    new_state,
                    success_count=state.success_count + 1,
                    half_open_in_flight=in_flight,
                )

            if await self.backend.compare_and_set(self.key, state, new_state):
                await self._trigger_hook(state, new_state)
                self.metrics.total_success += 1
                return

    async def record_failure(self) -> None:
        """Record a failed call and potentially open the circuit."""
        while True:
            state = await self._get_state()
            if state.status == BreakerStatus.OPEN:
                return

            new_state = add_result(state, False)
            ratio = failure_rating(new_state)
            in_flight = max(0, state.half_open_in_flight - 1)

            if ratio >= self.failure_threshold_ratio:
                new_state = evolve(
                    new_state,
                    status=BreakerStatus.OPEN,
                    failure_count=0,
                    success_count=0,
                    half_open_in_flight=in_flight,
                    opened_at=time.time(),
                )
            else:
                new_state = evolve(
                    new_state,
                    status=BreakerStatus.CLOSED,
                    failure_count=new_state.failure_count + 1,
                    success_count=0,
                    half_open_in_flight=in_flight,
                    opened_at=None,
                )

            if await self.backend.compare_and_set(self.key, state, new_state):
                await self._trigger_hook(state, new_state)
                self.metrics.total_failure += 1
                return

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    async def current_state(self) -> BreakerState:
        """Return the current :class:`~.state.BreakerState`."""
        return await self._get_state()

    async def failure_ratio(self) -> float:
        """Return the current failure ratio in the sliding window."""
        state = await self._get_state()
        return failure_rating(state)

    async def is_open(self) -> bool:
        """Return ``True`` if the circuit is currently OPEN."""
        state = await self._get_state()
        return state.status == BreakerStatus.OPEN

    def get_metrics(self) -> BreakerMetrics:
        """Return accumulated :class:`~.metrics.BreakerMetrics`."""
        return self.metrics
