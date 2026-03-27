"""
Groq provider adapter.

Wraps the Groq Cloud API via httpx. Groq exposes an OpenAI-compatible
Chat Completions endpoint, enabling ultra-low-latency inference on
open-source models (LLaMA 3, Mixtral, Gemma).

Supported models: llama-3-8b-8192, mixtral-8x7b, llama3-70b-8192.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from llm_std_lib.config import ProviderConfig
from llm_std_lib.exceptions import LLMProviderError, LLMRateLimitError, LLMTimeoutError
from llm_std_lib.providers.base import BaseProvider
from llm_std_lib.types import RequestContext, ResponseContext

_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
_DEFAULT_MODEL = "llama-3-8b-8192"


class GroqProvider(BaseProvider):
    """Adapter for the Groq Cloud API (OpenAI-compatible).

    Args:
        config: Provider config. ``api_key`` must be set (GROQ_API_KEY).
            ``base_url`` overrides the default endpoint.
    """

    name = "groq"

    def __init__(self, config: ProviderConfig) -> None:
        from llm_std_lib.exceptions import LLMConfigError

        if not config.api_key:
            raise LLMConfigError(
                "Groq api_key is required. Set GROQ_API_KEY or pass it in ProviderConfig."
            )
        self._api_key = config.api_key
        self._base_url = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._timeout = config.timeout_ms / 1000.0

    async def complete(self, ctx: RequestContext) -> ResponseContext:
        model = ctx.model or _DEFAULT_MODEL
        payload = self._build_payload(ctx, model, stream=False)
        start = time.monotonic()
        raw = await self._post("/chat/completions", payload)
        latency_ms = (time.monotonic() - start) * 1000
        return self._parse_response(ctx, raw, model, latency_ms)

    async def stream(self, ctx: RequestContext) -> AsyncGenerator[str, None]:  # pragma: no cover
        import json

        model = ctx.model or _DEFAULT_MODEL
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
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta

    # ------------------------------------------------------------------

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
                    f"{self._base_url}{path}", json=payload, headers=self._headers()
                )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Groq request timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(f"Groq connection error: {exc}") from exc
        self._raise_for_status(response.status_code, response.text)
        return response.json()  # type: ignore[no-any-return]

    @staticmethod
    def _raise_for_status(status_code: int, body: str = "") -> None:
        if status_code == 429:
            raise LLMRateLimitError(f"Groq rate limit exceeded (429). Body: {body}")
        if status_code >= 500:
            raise LLMProviderError(f"Groq server error ({status_code}). Body: {body}")
        if status_code >= 400:
            raise LLMProviderError(f"Groq client error ({status_code}). Body: {body}")

    def _parse_response(
        self, ctx: RequestContext, raw: dict[str, Any], model: str, latency_ms: float
    ) -> ResponseContext:
        text = raw["choices"][0]["message"]["content"] or ""
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
