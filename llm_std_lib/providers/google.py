"""
Google Gemini provider adapter.

Calls the Generative Language REST API (v1beta) via httpx.
Supports Gemini 1.5 Pro and Gemini 1.5 Flash (multimodal-capable).

Supported models: gemini-1.5-pro, gemini-1.5-flash.
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

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_MODEL = "gemini-1.5-flash"


class GoogleProvider(BaseProvider):
    """Adapter for the Google Generative Language API (Gemini).

    Args:
        config: Provider config. ``api_key`` must be set (GOOGLE_API_KEY).
            ``base_url`` overrides the default endpoint.
    """

    name = "google"

    def __init__(self, config: ProviderConfig) -> None:
        from llm_std_lib.exceptions import LLMConfigError

        if not config.api_key:
            raise LLMConfigError(
                "Google api_key is required. Set GOOGLE_API_KEY or pass it in ProviderConfig."
            )
        self._api_key = config.api_key
        self._base_url = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._timeout = config.timeout_ms / 1000.0

    async def complete(self, ctx: RequestContext) -> ResponseContext:
        model = ctx.model or _DEFAULT_MODEL
        payload = self._build_payload(ctx)
        url = f"{self._base_url}/models/{model}:generateContent?key={self._api_key}"
        start = time.monotonic()
        raw = await self._post(url, payload)
        latency_ms = (time.monotonic() - start) * 1000
        return self._parse_response(ctx, raw, model, latency_ms)

    async def stream(self, ctx: RequestContext) -> AsyncGenerator[str, None]:  # pragma: no cover
        import json

        model = ctx.model or _DEFAULT_MODEL
        payload = self._build_payload(ctx)
        url = (
            f"{self._base_url}/models/{model}:streamGenerateContent"
            f"?key={self._api_key}&alt=sse"
        )
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                self._raise_for_status(response.status_code)
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = json.loads(line[6:])
                    for candidate in chunk.get("candidates", []):
                        for part in candidate.get("content", {}).get("parts", []):
                            if text := part.get("text", ""):
                                yield text

    # ------------------------------------------------------------------

    def _build_payload(self, ctx: RequestContext) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        if ctx.system_prompt:
            contents.append({"role": "user", "parts": [{"text": ctx.system_prompt}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": ctx.prompt}]})

        payload: dict[str, Any] = {"contents": contents}
        gen_config: dict[str, Any] = {}
        if ctx.temperature is not None:
            gen_config["temperature"] = ctx.temperature
        if ctx.max_tokens is not None:
            gen_config["maxOutputTokens"] = ctx.max_tokens
        if gen_config:
            payload["generationConfig"] = gen_config
        return payload

    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    url, json=payload, headers={"Content-Type": "application/json"}
                )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Google request timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(f"Google connection error: {exc}") from exc
        self._raise_for_status(response.status_code, response.text)
        return response.json()  # type: ignore[no-any-return]

    @staticmethod
    def _raise_for_status(status_code: int, body: str = "") -> None:
        if status_code == 429:
            raise LLMRateLimitError(f"Google rate limit exceeded (429). Body: {body}")
        if status_code >= 500:
            raise LLMProviderError(f"Google server error ({status_code}). Body: {body}")
        if status_code >= 400:
            raise LLMProviderError(f"Google client error ({status_code}). Body: {body}")

    def _parse_response(
        self, ctx: RequestContext, raw: dict[str, Any], model: str, latency_ms: float
    ) -> ResponseContext:
        candidate = raw["candidates"][0]
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)

        usage = raw.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = usage.get("totalTokenCount", prompt_tokens + completion_tokens)
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
