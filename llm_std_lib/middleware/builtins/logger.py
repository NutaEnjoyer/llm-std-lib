"""
Structured request/response logging middleware.

Emits one ``llm.request`` event before dispatch and one ``llm.response``
(or ``llm.error``) event after, using structlog with all key metrics bound.
"""

from __future__ import annotations

from llm_std_lib._logging import get_logger
from llm_std_lib.middleware.base import BaseMiddleware
from llm_std_lib.types import RequestContext, ResponseContext

_log = get_logger(__name__)


class PromptLoggerMiddleware(BaseMiddleware):
    """Logs every LLM request and response via structlog.

    Args:
        log_prompt: Include the prompt text in the log event.
            Disable in production if prompts may contain sensitive data.
            Default: ``False``.
        log_response: Include the response text in the log event.
            Default: ``False``.

    Log events emitted:

    * ``llm.request``  — at DEBUG level, before the provider call.
    * ``llm.response`` — at INFO level, after a successful response.
    * ``llm.error``    — at WARNING level, when the provider raises.

    Example::

        stack = MiddlewareStack([PromptLoggerMiddleware(log_prompt=True)])
    """

    def __init__(
        self,
        log_prompt: bool = False,
        log_response: bool = False,
    ) -> None:
        self._log_prompt = log_prompt
        self._log_response = log_response

    async def pre_request(self, ctx: RequestContext) -> RequestContext:
        fields: dict[str, object] = {
            "request_id": ctx.request_id,
            "provider": ctx.provider,
            "model": ctx.model,
            "tags": ctx.tags,
        }
        if self._log_prompt:
            fields["prompt"] = ctx.prompt
        _log.debug("llm.request", **fields)
        return ctx

    async def post_request(
        self, ctx: RequestContext, response: ResponseContext
    ) -> ResponseContext:
        fields: dict[str, object] = {
            "request_id": response.request_id,
            "provider": response.provider,
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "cost_usd": response.cost_usd,
            "latency_ms": response.latency_ms,
            "cached": response.cached,
        }
        if self._log_response:
            fields["response"] = response.text
        _log.info("llm.response", **fields)
        return response

    async def on_error(self, ctx: RequestContext, exc: Exception) -> None:
        _log.warning(
            "llm.error",
            request_id=ctx.request_id,
            provider=ctx.provider,
            model=ctx.model,
            error=type(exc).__name__,
            detail=str(exc),
        )
