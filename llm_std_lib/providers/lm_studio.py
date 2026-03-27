"""
LM Studio provider adapter.

Wraps the LM Studio local HTTP server via httpx. LM Studio exposes an
OpenAI-compatible Chat Completions endpoint (default: ``http://localhost:1234``),
enabling local inference on GGUF models without any cloud dependency.

No API key is required. ``base_url`` defaults to ``http://localhost:1234``.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from llm_std_lib.config import ProviderConfig
from llm_std_lib.exceptions import LLMProviderError, LLMTimeoutError
from llm_std_lib.providers.base import BaseProvider
from llm_std_lib.types import RequestContext, ResponseContext

_DEFAULT_BASE_URL = "http://localhost:1234"
_DEFAULT_MODEL = "local-model"


class LMStudioProvider(BaseProvider):
    """Adapter for LM Studio local inference server (OpenAI-compatible).

    Args:
        config: Provider config.
            - ``base_url``: LM Studio server URL (default ``http://localhost:1234``).
            - ``api_key``: Optional bearer token (LM Studio is unauthenticated by default).
    """

    name = "lm_studio"

    def __init__(self, config: ProviderConfig) -> None:
        self._base_url = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._api_key = config.api_key  # optional
        self._timeout = config.timeout_ms / 1000.0

    async def complete(self, ctx: RequestContext) -> ResponseContext:
        model = ctx.model or _DEFAULT_MODEL
        payload = self._build_payload(ctx, model, stream=False)
        start = time.monotonic()
        raw = await self._post("/v1/chat/completions", payload)
        latency_ms = (time.monotonic() - start) * 1000
        return self._parse_response(ctx, raw, model, latency_ms)

    async def stream(self, ctx: RequestContext) -> AsyncGenerator[str, None]:  # pragma: no cover
        import json

        model = ctx.model or _DEFAULT_MODEL
        payload = self._build_payload(ctx, model, stream=True)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
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
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

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
            raise LLMTimeoutError(f"LM Studio request timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(
                f"LM Studio connection error: {exc}. Is LM Studio running at {self._base_url}?"
            ) from exc
        self._raise_for_status(response.status_code, response.text)
        return response.json()  # type: ignore[no-any-return]

    @staticmethod
    def _raise_for_status(status_code: int, body: str = "") -> None:
        if status_code >= 500:
            raise LLMProviderError(f"LM Studio server error ({status_code}). Body: {body}")
        if status_code >= 400:
            raise LLMProviderError(f"LM Studio client error ({status_code}). Body: {body}")

    def _parse_response(
        self, ctx: RequestContext, raw: dict[str, Any], model: str, latency_ms: float
    ) -> ResponseContext:
        text = raw["choices"][0]["message"]["content"] or ""
        usage = raw.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        return ResponseContext(
            request_id=ctx.request_id,
            text=text,
            model=model,
            provider=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=0.0,  # local inference — no cost
            latency_ms=latency_ms,
        )
