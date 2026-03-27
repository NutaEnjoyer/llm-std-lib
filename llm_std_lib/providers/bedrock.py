"""
AWS Bedrock provider adapter.

Wraps the AWS Bedrock Runtime API via boto3 (``bedrock-runtime`` client).
Runs boto3 calls in a thread-pool executor so the async interface stays
non-blocking.

Requires the ``boto3`` package (not bundled — install separately or via
``pip install boto3``).

Supported model families: Anthropic Claude, Amazon Titan.
Model IDs follow the Bedrock format: ``anthropic.claude-3-haiku-20240307-v1:0``.

Required ``extra`` keys in ProviderConfig (all optional):
    - ``region``: AWS region (default ``"us-east-1"``).
    - ``profile``: AWS profile name (uses default chain if omitted).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any

from llm_std_lib.config import ProviderConfig
from llm_std_lib.exceptions import LLMProviderError, LLMRateLimitError, LLMTimeoutError
from llm_std_lib.providers.base import BaseProvider
from llm_std_lib.types import RequestContext, ResponseContext

_DEFAULT_REGION = "us-east-1"
_DEFAULT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
_ANTHROPIC_VERSION = "bedrock-2023-05-31"


def _import_boto3() -> Any:
    try:
        import boto3

        return boto3
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for BedrockProvider. Install it with: pip install boto3"
        ) from exc


class BedrockProvider(BaseProvider):
    """Adapter for AWS Bedrock Runtime.

    Args:
        config: Provider config.
            - ``extra["region"]``: AWS region (default ``"us-east-1"``).
            - ``extra["profile"]``: AWS named profile (optional).
            - ``api_key`` is ignored — credentials come from the standard AWS
              credential chain (env vars, ``~/.aws/credentials``, IAM role).
    """

    name = "bedrock"

    def __init__(self, config: ProviderConfig) -> None:
        self._region = config.extra.get("region", _DEFAULT_REGION)
        self._profile = config.extra.get("profile")
        self._timeout = config.timeout_ms / 1000.0
        self._client: Any = None  # lazy-initialised

    def _get_client(self) -> Any:
        if self._client is None:
            boto3 = _import_boto3()
            session_kwargs: dict[str, Any] = {}
            if self._profile:
                session_kwargs["profile_name"] = self._profile
            session = boto3.Session(**session_kwargs)
            self._client = session.client(
                "bedrock-runtime",
                region_name=self._region,
            )
        return self._client

    async def complete(self, ctx: RequestContext) -> ResponseContext:
        model_id = ctx.model or _DEFAULT_MODEL
        body = self._build_body(ctx, model_id)
        start = time.monotonic()

        loop = asyncio.get_event_loop()
        try:
            raw = await asyncio.wait_for(
                loop.run_in_executor(None, self._invoke, model_id, body),
                timeout=self._timeout,
            )
        except TimeoutError as exc:
            raise LLMTimeoutError(f"Bedrock request timed out after {self._timeout}s") from exc

        latency_ms = (time.monotonic() - start) * 1000
        return self._parse_response(ctx, raw, model_id, latency_ms)

    async def stream(self, ctx: RequestContext) -> AsyncGenerator[str, None]:  # pragma: no cover
        model_id = ctx.model or _DEFAULT_MODEL
        body = self._build_body(ctx, model_id)
        loop = asyncio.get_event_loop()

        def _invoke_stream() -> Any:
            client = self._get_client()
            return client.invoke_model_with_response_stream(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )

        response = await loop.run_in_executor(None, _invoke_stream)
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if _is_anthropic(model_id):
                if chunk.get("type") == "content_block_delta":
                    yield chunk.get("delta", {}).get("text", "")
            else:
                # Amazon Titan
                yield chunk.get("outputText", "")

    # ------------------------------------------------------------------

    def _invoke(self, model_id: str, body: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        try:
            resp = client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
        except Exception as exc:
            name = type(exc).__name__
            msg = str(exc)
            if "ThrottlingException" in name or "TooManyRequests" in msg:
                raise LLMRateLimitError(f"Bedrock rate limit: {exc}") from exc
            raise LLMProviderError(f"Bedrock error: {exc}") from exc
        return json.loads(resp["body"].read())  # type: ignore[no-any-return]

    def _build_body(self, ctx: RequestContext, model_id: str) -> dict[str, Any]:
        if _is_anthropic(model_id):
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": ctx.prompt}
            ]
            body: dict[str, Any] = {
                "anthropic_version": _ANTHROPIC_VERSION,
                "messages": messages,
                "max_tokens": ctx.max_tokens or 1024,
            }
            if ctx.system_prompt:
                body["system"] = ctx.system_prompt
            if ctx.temperature is not None:
                body["temperature"] = ctx.temperature
        else:
            # Amazon Titan format
            body = {
                "inputText": (
                    f"{ctx.system_prompt}\n\n{ctx.prompt}"
                    if ctx.system_prompt
                    else ctx.prompt
                ),
                "textGenerationConfig": {
                    "maxTokenCount": ctx.max_tokens or 512,
                    "temperature": ctx.temperature if ctx.temperature is not None else 0.7,
                },
            }
        return body

    def _parse_response(
        self, ctx: RequestContext, raw: dict[str, Any], model_id: str, latency_ms: float
    ) -> ResponseContext:
        if _is_anthropic(model_id):
            text = raw["content"][0]["text"]
            usage = raw.get("usage", {})
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)
        else:
            # Amazon Titan
            text = raw["results"][0]["outputText"]
            prompt_tokens = raw.get("inputTextTokenCount", 0)
            completion_tokens = raw["results"][0].get("tokenCount", 0)

        total_tokens = prompt_tokens + completion_tokens

        return ResponseContext(
            request_id=ctx.request_id,
            text=text,
            model=model_id,
            provider=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=0.0,  # Bedrock pricing varies by account/region
            latency_ms=latency_ms,
        )


def _is_anthropic(model_id: str) -> bool:
    return model_id.startswith("anthropic.")
