"""
Base middleware interface.

All middleware — built-in and user-supplied — subclass :class:`BaseMiddleware`
and override whichever hooks they need.  Unoverridden hooks are no-ops, so a
middleware that only cares about responses only needs to implement
:meth:`post_request`.

Lifecycle per request::

    pre_request  →  provider call  →  post_request
                          ↓ (on exception)
                       on_error

Example — a minimal logging middleware::

    class TimingMiddleware(BaseMiddleware):
        async def post_request(
            self, ctx: RequestContext, response: ResponseContext
        ) -> ResponseContext:
            print(f"latency={response.latency_ms:.0f}ms")
            return response
"""

from __future__ import annotations

from llm_std_lib.types import RequestContext, ResponseContext


class BaseMiddleware:
    """Abstract base for all middleware components.

    Override any combination of the three hooks.  All hooks are async and
    receive the full request / response context.
    """

    async def pre_request(self, ctx: RequestContext) -> RequestContext:
        """Called before the request is dispatched to the provider.

        Args:
            ctx: Mutable request context.

        Returns:
            The (possibly modified) :class:`~llm_std_lib.types.RequestContext`.
        """
        return ctx

    async def post_request(
        self, ctx: RequestContext, response: ResponseContext
    ) -> ResponseContext:
        """Called after a successful provider response.

        Args:
            ctx: The original request context.
            response: Mutable response context.

        Returns:
            The (possibly modified) :class:`~llm_std_lib.types.ResponseContext`.
        """
        return response

    async def on_error(self, ctx: RequestContext, exc: Exception) -> None:
        """Called when the provider call raises an exception.

        The exception is *not* suppressed; this hook is for side-effects only
        (logging, metrics, alerting).

        Args:
            ctx: The request context at the time of the error.
            exc: The exception that was raised.
        """
