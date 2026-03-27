"""
Provider fallback chain.

FallbackChain wraps a list of LLM providers with per-provider circuit
breakers and retry policies. Providers are tried in order; the chain moves
to the next provider whenever the current one raises an LLMError or its
circuit breaker is OPEN. When every provider is exhausted,
:class:`~llm_std_lib.exceptions.LLMAllFallbacksFailedError` is raised.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable

from llm_std_lib.exceptions import LLMAllFallbacksFailedError, LLMError
from llm_std_lib.providers.base import BaseProvider
from llm_std_lib.types import RequestContext, ResponseContext

from ._exceptions import CircuitOpenError, MaxRetriesExceeded
from .backend import InMemoryBackend
from .circuit_breaker import CircuitBreaker
from .engine import ResilienceEngine
from .retry import RetryPolicy

logger = logging.getLogger("llm_std_lib.resilience.fallback")


class FallbackChain:
    """Tries a sequence of providers in order, falling back on failure.

    Each provider gets its own :class:`~.engine.ResilienceEngine` (circuit
    breaker + retry). Providers whose circuit is OPEN are skipped silently.
    The chain raises :class:`~llm_std_lib.exceptions.LLMAllFallbacksFailedError`
    only when every provider in the list has been exhausted.

    Args:
        providers: Ordered list of :class:`~llm_std_lib.providers.base.BaseProvider`
            instances. Tried left-to-right on each request.
        retry_policy: Shared :class:`~.retry.RetryPolicy` applied per provider.
            Defaults to ``RetryPolicy(max_attempts=2)``.
        failure_threshold_ratio: Failure ratio that opens a circuit breaker.
            Default: ``0.5``.
        recovery_timeout: Seconds the circuit stays OPEN before probing.
            Default: ``30.0``.

    Example::

        chain = FallbackChain(
            providers=[openai_provider, anthropic_provider, groq_provider],
            retry_policy=RetryPolicy(max_attempts=2, base_delay=0.3),
        )
        response = await chain.complete(ctx)
    """

    def __init__(
        self,
        providers: list[BaseProvider],
        retry_policy: RetryPolicy | None = None,
        failure_threshold_ratio: float = 0.5,
        recovery_timeout: float = 30.0,
    ) -> None:
        if not providers:
            raise ValueError("FallbackChain requires at least one provider.")

        policy = retry_policy or RetryPolicy(max_attempts=2)
        self._providers = list(providers)
        self._engines: dict[str, ResilienceEngine] = {}

        for provider in self._providers:
            backend = InMemoryBackend()
            breaker = CircuitBreaker(
                backend=backend,
                key=provider.name,
                failure_threshold_ratio=failure_threshold_ratio,
                recovery_timeout=recovery_timeout,
            )
            self._engines[provider.name] = ResilienceEngine(breaker, retryer=policy)

    async def complete(self, ctx: RequestContext) -> ResponseContext:
        """Execute the request through the chain.

        Args:
            ctx: The request context. ``ctx.provider`` and ``ctx.model`` are
                updated to reflect the provider that ultimately served the request.

        Returns:
            :class:`~llm_std_lib.types.ResponseContext` from the first
            successful provider.

        Raises:
            LLMAllFallbacksFailedError: When every provider has failed.
        """
        errors: list[Exception] = []

        for provider in self._providers:
            engine = self._engines[provider.name]

            # Skip providers whose circuit is currently OPEN
            if await engine.is_open():
                logger.debug(
                    "Skipping provider '%s' — circuit is OPEN", provider.name
                )
                errors.append(CircuitOpenError())
                continue

            try:
                # Stamp the context with the candidate provider
                ctx.provider = provider.name

                def _call(  # noqa: E731
                    p: BaseProvider = provider,
                    c: RequestContext = ctx,
                ) -> Awaitable[ResponseContext]:
                    return p.complete(c)

                result: ResponseContext = await engine.execute(_call)
                logger.debug("FallbackChain: served by '%s'", provider.name)
                return result

            except CircuitOpenError as exc:
                logger.debug(
                    "Provider '%s' circuit opened mid-request", provider.name
                )
                errors.append(exc)

            except MaxRetriesExceeded as exc:
                logger.warning(
                    "Provider '%s' exhausted retries", provider.name
                )
                errors.append(exc)

            except LLMError as exc:
                logger.warning(
                    "Provider '%s' raised %s, trying next",
                    provider.name,
                    type(exc).__name__,
                )
                errors.append(exc)

        raise LLMAllFallbacksFailedError(
            f"All {len(self._providers)} provider(s) in the fallback chain failed. "
            f"Errors: {[type(e).__name__ for e in errors]}"
        )

    def engine_for(self, provider_name: str) -> ResilienceEngine:
        """Return the :class:`~.engine.ResilienceEngine` for a provider.

        Useful for inspecting breaker state or injecting recorded latencies
        in tests.

        Args:
            provider_name: The provider's ``name`` attribute.

        Raises:
            KeyError: If the provider name is not in the chain.
        """
        return self._engines[provider_name]

    @property
    def providers(self) -> list[BaseProvider]:
        """The ordered list of providers in the chain."""
        return list(self._providers)
