"""
Azure OpenAI provider adapter.

Wraps the Azure OpenAI Service REST API via httpx. Uses the same Chat
Completions format as OpenAI but routes to a deployment-specific endpoint
and authenticates via ``api-key`` header.

Required ``extra`` keys in ProviderConfig:
    - ``deployment``: Azure deployment name (e.g. ``"gpt-4o-mini"``).
    - ``api_version``: Azure API version (default ``"2024-02-01"``).

``base_url`` must be the Azure resource endpoint:
``https://{resource}.openai.azure.com``.
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

_DEFAULT_API_VERSION = "2024-02-01"


class AzureProvider(BaseProvider):
    """Adapter for Azure OpenAI Service.

    Args:
        config: Provider config.
            - ``api_key``: Azure API key (or use Managed Identity — set to ``"managed_identity"``).
            - ``base_url``: Azure endpoint (``https://{resource}.openai.azure.com``).
            - ``extra["deployment"]``: Deployment name.
            - ``extra["api_version"]``: API version (default ``2024-02-01``).
    """

    name = "azure"

    def __init__(self, config: ProviderConfig) -> None:
        from llm_std_lib.exceptions import LLMConfigError

        if not config.base_url:
            raise LLMConfigError(
                "Azure base_url is required (e.g. https://{resource}.openai.azure.com). "
                "Pass it in ProviderConfig."
            )
        if not config.extra.get("deployment"):
            raise LLMConfigError(
                "Azure deployment name is required. "
                "Set ProviderConfig(extra={'deployment': 'your-deployment'})."
            )
        self._api_key = config.api_key
        self._base_url = config.base_url.rstrip("/")
        self._deployment = config.extra["deployment"]
        self._api_version = config.extra.get("api_version", _DEFAULT_API_VERSION)
        self._timeout = config.timeout_ms / 1000.0

    async def complete(self, ctx: RequestContext) -> ResponseContext:
        payload = self._build_payload(ctx, stream=False)
        start = time.monotonic()
        raw = await self._post(payload)
        latency_ms = (time.monotonic() - start) * 1000
        return self._parse_response(ctx, raw, latency_ms)

    async def stream(self, ctx: RequestContext) -> AsyncGenerator[str, None]:  # pragma: no cover
        import json

        payload = self._build_payload(ctx, stream=True)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST", self._endpoint(), json=payload, headers=self._headers()
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

    def _endpoint(self) -> str:
        return (
            f"{self._base_url}/openai/deployments/{self._deployment}"
            f"/chat/completions?api-version={self._api_version}"
        )

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key and self._api_key != "managed_identity":
            headers["api-key"] = self._api_key
        return headers

    def _build_payload(self, ctx: RequestContext, stream: bool) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if ctx.system_prompt:
            messages.append({"role": "system", "content": ctx.system_prompt})
        messages.append({"role": "user", "content": ctx.prompt})

        payload: dict[str, Any] = {"messages": messages, "stream": stream}
        if ctx.temperature is not None:
            payload["temperature"] = ctx.temperature
        if ctx.max_tokens is not None:
            payload["max_tokens"] = ctx.max_tokens
        return payload

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._endpoint(), json=payload, headers=self._headers()
                )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Azure request timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise LLMProviderError(f"Azure connection error: {exc}") from exc
        self._raise_for_status(response.status_code, response.text)
        return response.json()  # type: ignore[no-any-return]

    @staticmethod
    def _raise_for_status(status_code: int, body: str = "") -> None:
        if status_code == 429:
            raise LLMRateLimitError(f"Azure rate limit exceeded (429). Body: {body}")
        if status_code >= 500:
            raise LLMProviderError(f"Azure server error ({status_code}). Body: {body}")
        if status_code >= 400:
            raise LLMProviderError(f"Azure client error ({status_code}). Body: {body}")

    def _parse_response(
        self, ctx: RequestContext, raw: dict[str, Any], latency_ms: float
    ) -> ResponseContext:
        choice = raw["choices"][0]
        text = choice["message"]["content"] or ""
        usage = raw.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        # Azure billing matches OpenAI prices for the same model family
        cost = self.calculate_cost(self._deployment, prompt_tokens, completion_tokens)

        return ResponseContext(
            request_id=ctx.request_id,
            text=text,
            model=self._deployment,
            provider=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
