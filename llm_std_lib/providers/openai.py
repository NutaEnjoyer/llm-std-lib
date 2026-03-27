"""
OpenAI provider adapter.

Wraps the OpenAI Chat Completions API via httpx (no openai SDK dependency)
and translates between the library's normalised types and OpenAI's format.

Supported models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo.
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

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIProvider(BaseProvider):
    """Adapter for OpenAI Chat Completions API.

    Args:
        config: Provider-level configuration (api_key, timeout_ms, etc.).

    Raises:
        LLMConfigError: If api_key is not provided.
    """

    name = "openai"

    def __init__(self, config: ProviderConfig) -> None:
        from llm_std_lib.exceptions import LLMConfigError

        if not config.api_key:
            raise LLMConfigError(
                "OpenAI api_key is required. Set OPENAI_API_KEY or pass it in ProviderConfig."
            )
        self._api_key = config.api_key
        self._base_url = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._timeout = config.timeout_ms / 1000.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def complete(self, ctx: RequestContext) -> ResponseContext:
        """Send a non-streaming Chat Completions request.

        Args:
            ctx: Request context with prompt, model, and optional parameters.

        Returns:
            ResponseContext populated with the model's reply, token counts,
            cost estimate, and latency.

        Raises:
            LLMRateLimitError: On HTTP 429.
            LLMTimeoutError: On request timeout.
            LLMProviderError: On any other provider error.
        """
        model = self._resolve_model(ctx)
        payload = self._build_payload(ctx, model, stream=False)
        start = time.monotonic()

        raw = await self._post("/chat/completions", payload)

        latency_ms = (time.monotonic() - start) * 1000
        return self._parse_response(ctx, raw, model, latency_ms)

    async def stream(self, ctx: RequestContext) -> AsyncGenerator[str, None]:  # pragma: no cover
        """Send a streaming Chat Completions request and yield text chunks.

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
                f"{self._base_url}/chat/completions",
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
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_model(self, ctx: RequestContext) -> str:
        return ctx.model or _DEFAULT_MODEL

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, ctx: RequestContext, model: str, stream: bool) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if ctx.system_prompt:
            messages.append({"role": "system", "content": ctx.system_prompt})
        messages.append({"role": "user", "content": ctx.prompt})

        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if ctx.temperature is not None:
            payload["temperature"] = ctx.temperature
        if ctx.max_tokens is not None:
            payload["max_tokens"] = ctx.max_tokens
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
            raise LLMTimeoutError(f"OpenAI request timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(f"OpenAI connection error: {exc}") from exc

        self._raise_for_status(response.status_code, response.text)
        return response.json()  # type: ignore[no-any-return]

    @staticmethod
    def _raise_for_status(status_code: int, body: str = "") -> None:
        if status_code == 429:
            raise LLMRateLimitError(f"OpenAI rate limit exceeded (429). Body: {body}")
        if status_code >= 500:
            raise LLMProviderError(f"OpenAI server error ({status_code}). Body: {body}")
        if status_code >= 400:
            raise LLMProviderError(f"OpenAI client error ({status_code}). Body: {body}")

    def _parse_response(
        self,
        ctx: RequestContext,
        raw: dict[str, Any],
        model: str,
        latency_ms: float,
    ) -> ResponseContext:
        choice = raw["choices"][0]
        text = choice["message"]["content"] or ""
        usage = raw.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
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
