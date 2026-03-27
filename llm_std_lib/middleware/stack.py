"""
Middleware stack — executes an ordered chain of middleware around a provider call.

Execution order (onion model):

    mw[0].pre_request → mw[1].pre_request → ... → handler
                                                       ↓
    mw[0].post_request ← mw[1].post_request ← ...  ← response

On error, ``on_error`` is called on every middleware in *reverse* order, then
the exception is re-raised.

Usage::

    stack = MiddlewareStack([
        PIIRedactorMiddleware(),
        PromptLoggerMiddleware(),
        CostTrackerMiddleware(),
    ])

    response = await stack.execute(ctx, provider.complete)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from llm_std_lib.types import RequestContext, ResponseContext

from .base import BaseMiddleware


class MiddlewareStack:
    """Ordered collection of :class:`~.base.BaseMiddleware` instances.

    Args:
        middlewares: Initial list of middlewares, applied left-to-right on
            ``pre_request`` and right-to-left on ``post_request`` / ``on_error``.

    Example::

        stack = MiddlewareStack([PIIRedactorMiddleware(), CostTrackerMiddleware()])
        response = await stack.execute(ctx, provider.complete)
    """

    def __init__(self, middlewares: list[BaseMiddleware] | None = None) -> None:
        self._stack: list[BaseMiddleware] = list(middlewares or [])

    def add(self, middleware: BaseMiddleware) -> None:
        """Append *middleware* to the end of the stack."""
        self._stack.append(middleware)

    @property
    def middlewares(self) -> list[BaseMiddleware]:
        """Read-only view of the current middleware list."""
        return list(self._stack)

    async def execute(
        self,
        ctx: RequestContext,
        handler: Callable[[RequestContext], Awaitable[ResponseContext]],
    ) -> ResponseContext:
        """Run *handler* wrapped by all middleware hooks.

        Args:
            ctx: Initial request context; may be mutated by ``pre_request`` hooks.
            handler: Async callable that performs the actual provider call.

        Returns:
            The :class:`~llm_std_lib.types.ResponseContext` after all
            ``post_request`` hooks have run.

        Raises:
            Any exception raised by the handler or a middleware hook.
        """
        # pre_request: left → right
        for mw in self._stack:
            ctx = await mw.pre_request(ctx)

        try:
            response = await handler(ctx)
        except Exception as exc:
            # on_error: right → left
            for mw in reversed(self._stack):
                await mw.on_error(ctx, exc)
            raise

        # post_request: right → left
        for mw in reversed(self._stack):
            response = await mw.post_request(ctx, response)

        return response
