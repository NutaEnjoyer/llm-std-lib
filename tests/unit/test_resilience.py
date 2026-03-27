"""Unit tests for the resilience layer (v0.4.0).

Covers: RetryPolicy, CircuitBreaker, ResilienceEngine,
        TokenBucketLimiter, FallbackChain.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_std_lib.exceptions import LLMAllFallbacksFailedError, LLMProviderError
from llm_std_lib.providers.base import BaseProvider
from llm_std_lib.resilience import (
    CircuitBreaker,
    InMemoryBackend,
    MaxRetriesExceeded,
    ResilienceEngine,
    RetryPolicy,
    TokenBucketLimiter,
)
from llm_std_lib.resilience._exceptions import CircuitOpenError, RateLimitExceeded
from llm_std_lib.resilience._types import BreakerStatus
from llm_std_lib.resilience.fallback import FallbackChain
from llm_std_lib.types import RequestContext, ResponseContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(prompt: str = "hello") -> RequestContext:
    return RequestContext(prompt=prompt)


def _response(provider: str = "openai", model: str = "gpt-4o-mini") -> ResponseContext:
    return ResponseContext(
        request_id="test",
        text="ok",
        model=model,
        provider=provider,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )


class _FakeProvider(BaseProvider):
    """Minimal provider stub for testing."""

    def __init__(self, name: str, side_effect=None, response=None):
        self.name = name
        self._side_effect = side_effect
        self._response = response or _response(provider=name)
        self.call_count = 0

    async def complete(self, ctx: RequestContext) -> ResponseContext:
        self.call_count += 1
        if self._side_effect:
            raise self._side_effect
        return self._response

    async def stream(self, ctx: RequestContext) -> AsyncIterator[str]:  # pragma: no cover
        yield "ok"


def _engine(
    failure_threshold: float = 0.5,
    max_attempts: int = 0,
    recovery_timeout: float = 9999.0,
) -> ResilienceEngine:
    backend = InMemoryBackend()
    breaker = CircuitBreaker(
        backend,
        key="test",
        failure_threshold_ratio=failure_threshold,
        recovery_timeout=recovery_timeout,
    )
    policy = RetryPolicy(max_attempts=max_attempts, jitter=False)
    return ResilienceEngine(breaker, retryer=policy)


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_is_retryable_with_tuple(self) -> None:
        policy = RetryPolicy(retryable=(ValueError, TypeError))
        assert policy.is_retryable(ValueError("x")) is True
        assert policy.is_retryable(TypeError("x")) is True
        assert policy.is_retryable(RuntimeError("x")) is False

    def test_is_retryable_with_callable(self) -> None:
        policy = RetryPolicy(retryable=lambda e: isinstance(e, OSError))
        assert policy.is_retryable(OSError()) is True
        assert policy.is_retryable(ValueError()) is False

    def test_compute_delay_exponential(self) -> None:
        policy = RetryPolicy(base_delay=1.0, exponential=True, jitter=False)
        # delay = min(base * 2^attempt * 0.5, max_delay)
        assert policy._compute_delay(0) == pytest.approx(0.5, rel=1e-6)
        assert policy._compute_delay(1) == pytest.approx(1.0, rel=1e-6)
        assert policy._compute_delay(2) == pytest.approx(2.0, rel=1e-6)

    def test_compute_delay_linear(self) -> None:
        policy = RetryPolicy(base_delay=1.0, exponential=False, jitter=False)
        assert policy._compute_delay(0) == pytest.approx(0.5, rel=1e-6)
        assert policy._compute_delay(5) == pytest.approx(0.5, rel=1e-6)

    def test_compute_delay_capped_at_max(self) -> None:
        policy = RetryPolicy(base_delay=100.0, max_delay=5.0, exponential=True, jitter=False)
        assert policy._compute_delay(10) == pytest.approx(5.0, rel=1e-6)

    def test_compute_delay_jitter_in_range(self) -> None:
        policy = RetryPolicy(base_delay=1.0, exponential=False, jitter=True)
        for _ in range(30):
            d = policy._compute_delay(0)
            assert 0.0 <= d <= 0.5

    async def test_wait_calls_sleep(self) -> None:
        policy = RetryPolicy(base_delay=0.001, exponential=False, jitter=False)
        # Should not raise and should complete quickly
        await policy.wait(0)


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def _breaker(self, threshold: float = 0.5, recovery_timeout: float = 9999.0):
        backend = InMemoryBackend()
        return CircuitBreaker(
            backend,
            key="svc",
            failure_threshold_ratio=threshold,
            recovery_timeout=recovery_timeout,
        )

    async def test_starts_closed(self) -> None:
        breaker = self._breaker()
        assert await breaker.is_open() is False

    async def test_failure_opens_circuit(self) -> None:
        breaker = self._breaker(threshold=0.5)
        await breaker.before_call()
        await breaker.record_failure()
        # 1 failure / 1 total = 1.0 >= 0.5 → OPEN
        assert await breaker.is_open() is True

    async def test_open_circuit_raises(self) -> None:
        breaker = self._breaker(threshold=0.5)
        await breaker.before_call()
        await breaker.record_failure()
        with pytest.raises(CircuitOpenError):
            await breaker.before_call()

    async def test_success_keeps_closed(self) -> None:
        breaker = self._breaker()
        await breaker.before_call()
        await breaker.record_success()
        assert await breaker.is_open() is False

    async def test_mixed_below_threshold_stays_closed(self) -> None:
        breaker = self._breaker(threshold=0.6)
        # Record 4 successes then 1 failure → 1/5 = 0.2 < 0.6
        for _ in range(4):
            await breaker.before_call()
            await breaker.record_success()
        await breaker.before_call()
        await breaker.record_failure()
        assert await breaker.is_open() is False

    async def test_failure_ratio(self) -> None:
        breaker = self._breaker(threshold=0.9)
        for _ in range(4):
            await breaker.before_call()
            await breaker.record_success()
        await breaker.before_call()
        await breaker.record_failure()
        ratio = await breaker.failure_ratio()
        assert ratio == pytest.approx(0.2, rel=1e-6)

    async def test_current_state(self) -> None:
        breaker = self._breaker()
        state = await breaker.current_state()
        assert state.status == BreakerStatus.CLOSED

    async def test_metrics_count(self) -> None:
        breaker = self._breaker(threshold=0.9)
        for _ in range(3):
            await breaker.before_call()
            await breaker.record_success()
        m = breaker.get_metrics()
        assert m.total_calls == 3
        assert m.total_success == 3
        assert m.total_failure == 0

    async def test_on_open_hook_called(self) -> None:
        hook = AsyncMock()
        backend = InMemoryBackend()
        breaker = CircuitBreaker(
            backend, key="svc", failure_threshold_ratio=0.5, on_open=hook
        )
        await breaker.before_call()
        await breaker.record_failure()
        hook.assert_awaited_once()

    async def test_on_close_hook_called_on_recovery(self) -> None:
        hook_close = AsyncMock()
        backend = InMemoryBackend()
        breaker = CircuitBreaker(
            backend,
            key="svc",
            failure_threshold_ratio=0.5,
            recovery_timeout=0.0,  # instant recovery
            on_close=hook_close,
        )
        # Open the circuit
        await breaker.before_call()
        await breaker.record_failure()
        assert await breaker.is_open() is True

        # With recovery_timeout=0.0 the next before_call moves to HALF_OPEN
        await breaker.before_call()
        # Now record enough successes to close
        state = await breaker.current_state()
        required = state.half_open_required_successes
        for _ in range(required):
            await breaker.record_success()

        hook_close.assert_awaited_once()


# ---------------------------------------------------------------------------
# ResilienceEngine
# ---------------------------------------------------------------------------


async def _pre_seed(engine: ResilienceEngine, successes: int = 18) -> None:
    """Fill the circuit breaker window with successes so early failures don't open it."""
    for _ in range(successes):
        await engine._breaker.before_call()
        await engine._breaker.record_success()


class TestResilienceEngine:
    async def test_execute_success(self) -> None:
        engine = _engine()
        result = await engine.execute(AsyncMock(return_value=42))
        assert result == 42

    async def test_execute_retries_on_error(self) -> None:
        # Pre-seed window so 2 failures stay below the 0.5 threshold (2/20=0.1)
        engine = _engine(failure_threshold=0.5, max_attempts=2)
        await _pre_seed(engine)
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMProviderError("transient")
            return "ok"

        result = await engine.execute(flaky)
        assert result == "ok"
        assert call_count == 3  # initial + 2 retries

    async def test_execute_raises_max_retries_exceeded(self) -> None:
        # Pre-seed window so retries run without circuit opening mid-way
        engine = _engine(failure_threshold=0.5, max_attempts=2)
        await _pre_seed(engine)

        async def always_fail():
            raise LLMProviderError("fail")

        with pytest.raises(MaxRetriesExceeded):
            await engine.execute(always_fail)

    async def test_execute_raises_circuit_open(self) -> None:
        engine = _engine(failure_threshold=0.5, max_attempts=0)
        # First call: 1 failure in empty window (ratio=1.0>=0.5) → opens circuit,
        # then MaxRetriesExceeded is raised (max_attempts=0 so no retries)
        with pytest.raises(MaxRetriesExceeded):
            await engine.execute(AsyncMock(side_effect=LLMProviderError("fail")))
        # Next call: circuit is OPEN → CircuitOpenError immediately
        with pytest.raises(CircuitOpenError):
            await engine.execute(AsyncMock(return_value="ok"))

    async def test_execute_timeout(self) -> None:
        backend = InMemoryBackend()
        breaker = CircuitBreaker(backend, key="t", failure_threshold_ratio=0.9)
        policy = RetryPolicy(max_attempts=0)
        engine = ResilienceEngine(breaker, retryer=policy, timeout=0.01)

        async def slow():
            await asyncio.sleep(1)

        with pytest.raises(asyncio.TimeoutError):
            await engine.execute(slow)

    async def test_protect_as_decorator(self) -> None:
        engine = _engine()
        called = False

        @engine.protect
        async def handler():
            nonlocal called
            called = True
            return "done"

        result = await handler()
        assert result == "done"
        assert called is True

    async def test_protect_as_context_manager(self) -> None:
        engine = _engine()
        async with engine.protect() as ctx:
            pass  # no exception → record_success
        assert engine.breaker_metrics().total_success == 1

    async def test_retry_metrics(self) -> None:
        engine = _engine(failure_threshold=0.5, max_attempts=2)
        await _pre_seed(engine)
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMProviderError("err")
            return "ok"

        await engine.execute(flaky)
        assert engine.retry_metrics().total_retries == 2


# ---------------------------------------------------------------------------
# TokenBucketLimiter
# ---------------------------------------------------------------------------


class TestTokenBucketLimiter:
    async def test_acquire_within_capacity(self) -> None:
        limiter = TokenBucketLimiter(capacity=5, refill_rate=100.0)
        for _ in range(5):
            assert await limiter.acquire() is True

    async def test_acquire_fails_when_empty(self) -> None:
        limiter = TokenBucketLimiter(capacity=1, refill_rate=0.0)
        await limiter.acquire()  # consume the only token
        assert await limiter.acquire() is False

    async def test_refill_over_time(self) -> None:
        limiter = TokenBucketLimiter(capacity=10, refill_rate=1000.0)
        # drain all
        for _ in range(10):
            await limiter.acquire()
        assert await limiter.acquire() is False
        # Sleep long enough for Windows timer resolution (~15ms minimum)
        await asyncio.sleep(0.05)
        assert await limiter.acquire() is True

    async def test_limit_decorator_raises(self) -> None:
        limiter = TokenBucketLimiter(capacity=1, refill_rate=0.0)
        await limiter.acquire()  # drain

        @limiter.limit()
        async def handler():
            return "ok"

        with pytest.raises(RateLimitExceeded):
            await handler()

    async def test_limit_decorator_succeeds(self) -> None:
        limiter = TokenBucketLimiter(capacity=5, refill_rate=10.0)

        @limiter.limit()
        async def handler():
            return "ok"

        assert await handler() == "ok"


# ---------------------------------------------------------------------------
# FallbackChain
# ---------------------------------------------------------------------------


class TestFallbackChain:
    def test_empty_providers_raises(self) -> None:
        with pytest.raises(ValueError):
            FallbackChain(providers=[])

    async def test_first_provider_succeeds(self) -> None:
        p1 = _FakeProvider("openai")
        p2 = _FakeProvider("anthropic")
        chain = FallbackChain([p1, p2])

        ctx = _ctx()
        result = await chain.complete(ctx)

        assert result.text == "ok"
        assert p1.call_count == 1
        assert p2.call_count == 0

    async def test_falls_back_on_llm_error(self) -> None:
        p1 = _FakeProvider("openai", side_effect=LLMProviderError("down"))
        p2 = _FakeProvider("anthropic")
        chain = FallbackChain(
            [p1, p2],
            retry_policy=RetryPolicy(max_attempts=0),
            failure_threshold_ratio=0.9,  # keep p1 open only after many failures
        )

        ctx = _ctx()
        result = await chain.complete(ctx)

        assert result.provider == "anthropic"
        assert p2.call_count == 1

    async def test_all_fail_raises(self) -> None:
        p1 = _FakeProvider("openai", side_effect=LLMProviderError("fail"))
        p2 = _FakeProvider("anthropic", side_effect=LLMProviderError("fail"))
        chain = FallbackChain(
            [p1, p2],
            retry_policy=RetryPolicy(max_attempts=0),
            failure_threshold_ratio=0.9,
        )

        with pytest.raises(LLMAllFallbacksFailedError):
            await chain.complete(_ctx())

    async def test_skips_open_circuit(self) -> None:
        p1 = _FakeProvider("openai", side_effect=LLMProviderError("fail"))
        p2 = _FakeProvider("anthropic")
        chain = FallbackChain(
            [p1, p2],
            retry_policy=RetryPolicy(max_attempts=0),
            failure_threshold_ratio=0.5,
        )

        # Trigger p1 circuit open via a failed call
        try:
            await chain.complete(_ctx())
        except LLMAllFallbacksFailedError:
            pass

        p1.call_count = 0
        p2.call_count = 0

        # p1 circuit is now OPEN — chain should skip it directly
        result = await chain.complete(_ctx())
        assert result.provider == "anthropic"
        assert p1.call_count == 0  # skipped entirely
        assert p2.call_count == 1

    async def test_providers_property(self) -> None:
        p1 = _FakeProvider("openai")
        p2 = _FakeProvider("anthropic")
        chain = FallbackChain([p1, p2])
        assert chain.providers == [p1, p2]

    async def test_engine_for(self) -> None:
        p1 = _FakeProvider("openai")
        chain = FallbackChain([p1])
        engine = chain.engine_for("openai")
        assert isinstance(engine, ResilienceEngine)

    async def test_engine_for_unknown_raises(self) -> None:
        chain = FallbackChain([_FakeProvider("openai")])
        with pytest.raises(KeyError):
            chain.engine_for("unknown")

    async def test_falls_back_on_rate_limit_429(self) -> None:
        """Chaos: primary returns 429 → chain falls back to secondary."""
        from llm_std_lib.exceptions import LLMRateLimitError

        p1 = _FakeProvider("openai", side_effect=LLMRateLimitError("429 Too Many Requests"))
        p2 = _FakeProvider("anthropic")
        chain = FallbackChain(
            [p1, p2],
            retry_policy=RetryPolicy(max_attempts=0),
            failure_threshold_ratio=0.9,
        )
        result = await chain.complete(_ctx())
        assert result.provider == "anthropic"

    async def test_falls_back_on_timeout(self) -> None:
        """Chaos: primary times out → chain falls back to secondary."""
        from llm_std_lib.exceptions import LLMTimeoutError

        p1 = _FakeProvider("openai", side_effect=LLMTimeoutError("upstream timeout"))
        p2 = _FakeProvider("anthropic")
        chain = FallbackChain(
            [p1, p2],
            retry_policy=RetryPolicy(max_attempts=0),
            failure_threshold_ratio=0.9,
        )
        result = await chain.complete(_ctx())
        assert result.provider == "anthropic"

    async def test_e2e_primary_unavailable_uses_fallback(self) -> None:
        """E2E: primary circuit opens after failure, subsequent requests go directly to secondary."""
        p1 = _FakeProvider("openai", side_effect=LLMProviderError("down"))
        p2 = _FakeProvider("anthropic")
        chain = FallbackChain(
            [p1, p2],
            retry_policy=RetryPolicy(max_attempts=0),
            failure_threshold_ratio=0.5,
        )
        # First request: p1 fails → opens circuit → falls back to p2
        result1 = await chain.complete(_ctx())
        assert result1.provider == "anthropic"

        # Second request: p1 circuit is OPEN → skipped entirely, p2 serves directly
        p1.call_count = 0
        result2 = await chain.complete(_ctx())
        assert result2.provider == "anthropic"
        assert p1.call_count == 0  # never attempted

    async def test_three_provider_chain_uses_third(self) -> None:
        p1 = _FakeProvider("openai", side_effect=LLMProviderError("fail"))
        p2 = _FakeProvider("anthropic", side_effect=LLMProviderError("fail"))
        p3 = _FakeProvider("groq")
        chain = FallbackChain(
            [p1, p2, p3],
            retry_policy=RetryPolicy(max_attempts=0),
            failure_threshold_ratio=0.9,
        )
        result = await chain.complete(_ctx())
        assert result.provider == "groq"
        assert p3.call_count == 1
