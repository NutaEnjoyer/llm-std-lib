"""
Anthropic provider adapter.

Wraps the Anthropic Messages API (Claude models) via httpx and translates
between the library's normalised types and Anthropic's request/response format.

Supported models: claude-opus-3, claude-sonnet-3-5, claude-haiku-3.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from llm_std_lib.config import ProviderConfig
from llm_std_lib.exceptions import (
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from llm_std_lib.providers.base import BaseProvider
from llm_std_lib.types import RequestContext, ResponseContext

_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
_DEFAULT_MODEL = "claude-haiku-3"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    """Adapter for the Anthropic Messages API (Claude models).

    Args:
        config: Provider-level configuration (api_key, timeout_ms, etc.).

    Raises:
        LLMConfigError: If api_key is not provided.
    """

    name = "anthropic"

    def __init__(self, config: ProviderConfig) -> None:
        from llm_std_lib.exceptions import LLMConfigError

        if not config.api_key:
            raise LLMConfigError(
                "Anthropic api_key is required. Set ANTHROPIC_API_KEY or pass it in ProviderConfig."
            )
        self._api_key = config.api_key
        self._base_url = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._timeout = config.timeout_ms / 1000.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def complete(self, ctx: RequestContext) -> ResponseContext:
        """Send a non-streaming Messages request to Anthropic.

        Args:
            ctx: Request context with prompt, model, and optional parameters.

        Returns:
            ResponseContext populated with text, token counts, cost, and latency.

        Raises:
            LLMRateLimitError: On HTTP 429.
            LLMTimeoutError: On request timeout.
            LLMProviderError: On any other provider error.
        """
        model = self._resolve_model(ctx)
        payload = self._build_payload(ctx, model, stream=False)
        start = time.monotonic()

        raw = await self._post("/messages", payload)

        latency_ms = (time.monotonic() - start) * 1000
        return self._parse_response(ctx, raw, model, latency_ms)

    async def stream(self, ctx: RequestContext) -> AsyncGenerator[str, None]:  # pragma: no cover
        """Send a streaming Messages request and yield text chunks.

        Args:
            ctx: Request context with prompt, model, and optional parameters.

        Yields:
            Text delta strings as they arrive from the API.

        Raises:
            LLMRateLimitError: On HTTP 429.
            LLMTimeoutError: On request timeout.
            LLMProviderError: On any other provider error.
        """
        model = self._resolve_model(ctx)
        payload = self._build_payload(ctx, model, stream=True)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/messages",
                json=payload,
                headers=self._headers(),
            ) as response:
                self._raise_for_status(response.status_code)
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data.strip() == "[DONE]":
                        break
                    import json
                    event = json.loads(data)
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {}).get("text", "")
                        if delta:
                            yield delta

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_model(self, ctx: RequestContext) -> str:
        return ctx.model or _DEFAULT_MODEL

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def _build_payload(self, ctx: RequestContext, model: str, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": ctx.prompt}],
            "stream": stream,
        }
        if ctx.system_prompt:
            payload["system"] = ctx.system_prompt
        if ctx.max_tokens is not None:
            payload["max_tokens"] = ctx.max_tokens
        else:
            payload["max_tokens"] = 4096  # Anthropic requires max_tokens
        if ctx.temperature is not None:
            payload["temperature"] = ctx.temperature
        return payload

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}{path}",
                    json=payload,
                    headers=self._headers(),
                )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Anthropic request timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(f"Anthropic connection error: {exc}") from exc

        self._raise_for_status(response.status_code, response.text)
        return response.json()  # type: ignore[no-any-return]

    @staticmethod
    def _raise_for_status(status_code: int, body: str = "") -> None:
        if status_code == 429:
            raise LLMRateLimitError(f"Anthropic rate limit exceeded (429). Body: {body}")
        if status_code >= 500:
            raise LLMProviderError(f"Anthropic server error ({status_code}). Body: {body}")
        if status_code >= 400:
            raise LLMProviderError(f"Anthropic client error ({status_code}). Body: {body}")

    def _parse_response(
        self,
        ctx: RequestContext,
        raw: dict[str, Any],
        model: str,
        latency_ms: float,
    ) -> ResponseContext:
        # Anthropic returns content as a list of blocks
        content_blocks = raw.get("content", [])
        text = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )

        usage = raw.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)

        return ResponseContext(
            request_id=ctx.request_id,
            text=text,
            model=model,
            provider=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
