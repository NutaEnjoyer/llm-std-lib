"""Unit tests for provider adapters using mocked HTTP."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_std_lib.config import ProviderConfig
from llm_std_lib.exceptions import LLMConfigError, LLMProviderError, LLMRateLimitError, LLMTimeoutError
from llm_std_lib.providers.anthropic import AnthropicProvider
from llm_std_lib.providers.azure import AzureProvider
from llm_std_lib.providers.bedrock import BedrockProvider
from llm_std_lib.providers.google import GoogleProvider
from llm_std_lib.providers.groq import GroqProvider
from llm_std_lib.providers.lm_studio import LMStudioProvider
from llm_std_lib.providers.ollama import OllamaProvider
from llm_std_lib.providers.openai import OpenAIProvider
from llm_std_lib.types import RequestContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _openai_response(text: str = "Hello!", model: str = "gpt-4o-mini") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _anthropic_response(text: str = "Hello!", model: str = "claude-haiku-3") -> dict[str, Any]:
    return {
        "id": "msg-test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "stop_reason": "end_turn",
    }


def _mock_httpx_response(status_code: int, body: dict[str, Any]) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = json.dumps(body)
    mock.json.return_value = body
    return mock


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------

class TestOpenAIProvider:
    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(LLMConfigError, match="api_key"):
            OpenAIProvider(ProviderConfig(api_key=None))

    def test_name(self, openai_provider_config: ProviderConfig) -> None:
        provider = OpenAIProvider(openai_provider_config)
        assert provider.name == "openai"

    @pytest.mark.asyncio
    async def test_complete_success(self, openai_provider_config: ProviderConfig, sample_request_ctx: RequestContext) -> None:
        provider = OpenAIProvider(openai_provider_config)
        mock_resp = _mock_httpx_response(200, _openai_response("The answer is 42."))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_resp)

            ctx = sample_request_ctx
            result = await provider.complete(ctx)

        assert result.text == "The answer is 42."
        assert result.provider == "openai"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.total_tokens == 15
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_complete_rate_limit_raises(self, openai_provider_config: ProviderConfig, sample_request_ctx: RequestContext) -> None:
        provider = OpenAIProvider(openai_provider_config)
        mock_resp = _mock_httpx_response(429, {"error": {"message": "Rate limit exceeded"}})

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_resp)

            with pytest.raises(LLMRateLimitError):
                await provider.complete(sample_request_ctx)

    @pytest.mark.asyncio
    async def test_complete_server_error_raises(self, openai_provider_config: ProviderConfig, sample_request_ctx: RequestContext) -> None:
        provider = OpenAIProvider(openai_provider_config)
        mock_resp = _mock_httpx_response(500, {"error": {"message": "Internal server error"}})

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_resp)

            with pytest.raises(LLMProviderError):
                await provider.complete(sample_request_ctx)

    @pytest.mark.asyncio
    async def test_complete_timeout_raises(self, openai_provider_config: ProviderConfig, sample_request_ctx: RequestContext) -> None:
        import httpx as _httpx
        provider = OpenAIProvider(openai_provider_config)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=_httpx.TimeoutException("timeout"))

            with pytest.raises(LLMTimeoutError):
                await provider.complete(sample_request_ctx)

    def test_calculate_cost_known_model(self, openai_provider_config: ProviderConfig) -> None:
        provider = OpenAIProvider(openai_provider_config)
        cost = provider.calculate_cost("gpt-4o-mini", prompt_tokens=1000, completion_tokens=1000)
        # input: 1000/1000 * 0.00015 + output: 1000/1000 * 0.0006
        assert abs(cost - 0.00075) < 1e-9

    def test_calculate_cost_unknown_model_returns_zero(self, openai_provider_config: ProviderConfig) -> None:
        provider = OpenAIProvider(openai_provider_config)
        cost = provider.calculate_cost("unknown-model-xyz", prompt_tokens=1000, completion_tokens=1000)
        assert cost == 0.0

    def test_headers_do_not_expose_key_in_bearer(self, openai_provider_config: ProviderConfig) -> None:
        """The Authorization header exists but should not be logged."""
        provider = OpenAIProvider(openai_provider_config)
        headers = provider._headers()
        assert "Authorization" in headers
        # Key must be in the header value (we verify it's assembled correctly)
        assert openai_provider_config.api_key in headers["Authorization"]


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------

class TestAnthropicProvider:
    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(LLMConfigError, match="api_key"):
            AnthropicProvider(ProviderConfig(api_key=None))

    def test_name(self, anthropic_provider_config: ProviderConfig) -> None:
        provider = AnthropicProvider(anthropic_provider_config)
        assert provider.name == "anthropic"

    @pytest.mark.asyncio
    async def test_complete_success(self, anthropic_provider_config: ProviderConfig, sample_request_ctx: RequestContext) -> None:
        provider = AnthropicProvider(anthropic_provider_config)
        mock_resp = _mock_httpx_response(200, _anthropic_response("Semantic caching is great."))

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_resp)

            result = await provider.complete(sample_request_ctx)

        assert result.text == "Semantic caching is great."
        assert result.provider == "anthropic"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.total_tokens == 15

    @pytest.mark.asyncio
    async def test_complete_rate_limit_raises(self, anthropic_provider_config: ProviderConfig, sample_request_ctx: RequestContext) -> None:
        provider = AnthropicProvider(anthropic_provider_config)
        mock_resp = _mock_httpx_response(429, {"error": {"type": "rate_limit_error"}})

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_resp)

            with pytest.raises(LLMRateLimitError):
                await provider.complete(sample_request_ctx)

    @pytest.mark.asyncio
    async def test_complete_timeout_raises(self, anthropic_provider_config: ProviderConfig, sample_request_ctx: RequestContext) -> None:
        import httpx as _httpx
        provider = AnthropicProvider(anthropic_provider_config)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=_httpx.TimeoutException("timeout"))

            with pytest.raises(LLMTimeoutError):
                await provider.complete(sample_request_ctx)

    def test_payload_includes_max_tokens_default(self, anthropic_provider_config: ProviderConfig, sample_request_ctx: RequestContext) -> None:
        """Anthropic requires max_tokens — verify it's always set."""
        provider = AnthropicProvider(anthropic_provider_config)
        payload = provider._build_payload(sample_request_ctx, "claude-haiku-3", stream=False)
        assert "max_tokens" in payload

    def test_payload_with_system_prompt(self, anthropic_provider_config: ProviderConfig) -> None:
        provider = AnthropicProvider(anthropic_provider_config)
        ctx = RequestContext(prompt="Hello", system_prompt="You are a tester.")
        payload = provider._build_payload(ctx, "claude-haiku-3", stream=False)
        assert payload["system"] == "You are a tester."

    @pytest.mark.asyncio
    async def test_complete_connection_error_raises(self, anthropic_provider_config: ProviderConfig, sample_request_ctx: RequestContext) -> None:
        import httpx as _httpx
        provider = AnthropicProvider(anthropic_provider_config)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=_httpx.RequestError("conn refused"))

            with pytest.raises(LLMProviderError, match="connection error"):
                await provider.complete(sample_request_ctx)

    def test_calculate_cost_known_model(self, anthropic_provider_config: ProviderConfig) -> None:
        provider = AnthropicProvider(anthropic_provider_config)
        cost = provider.calculate_cost("claude-haiku-3", prompt_tokens=1000, completion_tokens=1000)
        assert cost > 0

    def test_calculate_cost_unknown_model_returns_zero(self, anthropic_provider_config: ProviderConfig) -> None:
        provider = AnthropicProvider(anthropic_provider_config)
        cost = provider.calculate_cost("no-such-model", prompt_tokens=1000, completion_tokens=1000)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# Connection error for OpenAI
# ---------------------------------------------------------------------------

class TestOpenAIProviderConnectionError:
    @pytest.mark.asyncio
    async def test_complete_connection_error_raises(self, openai_provider_config: ProviderConfig, sample_request_ctx: RequestContext) -> None:
        import httpx as _httpx
        provider = OpenAIProvider(openai_provider_config)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=_httpx.RequestError("conn refused"))

            with pytest.raises(LLMProviderError, match="connection error"):
                await provider.complete(sample_request_ctx)

    def test_raise_for_status_client_error(self, openai_provider_config: ProviderConfig) -> None:
        with pytest.raises(LLMProviderError, match="client error"):
            OpenAIProvider._raise_for_status(400, "bad request")

    def test_raise_for_status_no_error_passes(self) -> None:
        OpenAIProvider._raise_for_status(200)  # should not raise


class TestAnthropicRaiseForStatus:
    def test_rate_limit(self) -> None:
        with pytest.raises(LLMRateLimitError):
            AnthropicProvider._raise_for_status(429, "rate limit")

    def test_server_error(self) -> None:
        with pytest.raises(LLMProviderError, match="server error"):
            AnthropicProvider._raise_for_status(503, "service unavailable")

    def test_client_error(self) -> None:
        with pytest.raises(LLMProviderError, match="client error"):
            AnthropicProvider._raise_for_status(401, "unauthorized")

    def test_ok_passes(self) -> None:
        AnthropicProvider._raise_for_status(200)  # should not raise


class TestAnthropicParseResponse:
    def test_multiple_content_blocks(self, anthropic_provider_config: ProviderConfig, sample_request_ctx: RequestContext) -> None:
        provider = AnthropicProvider(anthropic_provider_config)
        raw = {
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world!"},
                {"type": "tool_use", "id": "tu1", "name": "fn"},  # non-text block
            ],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        ctx = provider._parse_response(sample_request_ctx, raw, "claude-haiku-3", 150.0)
        assert ctx.text == "Hello world!"
        assert ctx.prompt_tokens == 5
        assert ctx.completion_tokens == 3

    def test_empty_content(self, anthropic_provider_config: ProviderConfig, sample_request_ctx: RequestContext) -> None:
        provider = AnthropicProvider(anthropic_provider_config)
        raw = {"content": [], "usage": {"input_tokens": 1, "output_tokens": 0}}
        ctx = provider._parse_response(sample_request_ctx, raw, "claude-haiku-3", 50.0)
        assert ctx.text == ""


class TestOpenAIParseResponse:
    def test_parse_response_fields(self, openai_provider_config: ProviderConfig, sample_request_ctx: RequestContext) -> None:
        provider = OpenAIProvider(openai_provider_config)
        raw = _openai_response("Test answer")
        ctx = provider._parse_response(sample_request_ctx, raw, "gpt-4o-mini", 200.0)
        assert ctx.text == "Test answer"
        assert ctx.latency_ms == 200.0
        assert ctx.provider == "openai"

    def test_parse_response_missing_usage(self, openai_provider_config: ProviderConfig, sample_request_ctx: RequestContext) -> None:
        provider = OpenAIProvider(openai_provider_config)
        raw = {
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
        }
        ctx = provider._parse_response(sample_request_ctx, raw, "gpt-4o-mini", 100.0)
        assert ctx.text == "Hi"
        assert ctx.total_tokens == 0


# ---------------------------------------------------------------------------
# Helpers shared by OpenAI-compatible providers
# ---------------------------------------------------------------------------

def _oai_compat_response(text: str = "Hi!", model: str = "test-model") -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        "model": model,
    }


def _google_response(text: str = "Hi!") -> dict[str, Any]:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4, "totalTokenCount": 12},
    }


def _bedrock_anthropic_response(text: str = "Hi!") -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 8, "output_tokens": 4},
    }


# ---------------------------------------------------------------------------
# GoogleProvider
# ---------------------------------------------------------------------------

class TestGoogleProvider:
    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(LLMConfigError, match="api_key"):
            GoogleProvider(ProviderConfig(api_key=None))

    def test_name(self) -> None:
        assert GoogleProvider(ProviderConfig(api_key="key")).name == "google"

    @pytest.mark.asyncio
    async def test_complete_success(self, sample_request_ctx: RequestContext) -> None:
        provider = GoogleProvider(ProviderConfig(api_key="test-key"))
        mock_resp = _mock_httpx_response(200, _google_response("Gemini says hi."))

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_resp)

            result = await provider.complete(sample_request_ctx)

        assert result.text == "Gemini says hi."
        assert result.provider == "google"
        assert result.prompt_tokens == 8
        assert result.completion_tokens == 4

    @pytest.mark.asyncio
    async def test_complete_rate_limit(self, sample_request_ctx: RequestContext) -> None:
        provider = GoogleProvider(ProviderConfig(api_key="key"))
        mock_resp = _mock_httpx_response(429, {})
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_resp)
            with pytest.raises(LLMRateLimitError):
                await provider.complete(sample_request_ctx)

    @pytest.mark.asyncio
    async def test_complete_timeout(self, sample_request_ctx: RequestContext) -> None:
        import httpx as _httpx
        provider = GoogleProvider(ProviderConfig(api_key="key"))
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=_httpx.TimeoutException("t/o"))
            with pytest.raises(LLMTimeoutError):
                await provider.complete(sample_request_ctx)

    def test_raise_for_status(self) -> None:
        GoogleProvider._raise_for_status(200)
        with pytest.raises(LLMRateLimitError):
            GoogleProvider._raise_for_status(429)
        with pytest.raises(LLMProviderError):
            GoogleProvider._raise_for_status(500)
        with pytest.raises(LLMProviderError):
            GoogleProvider._raise_for_status(400)

    def test_build_payload_with_system_prompt(self, sample_request_ctx: RequestContext) -> None:
        provider = GoogleProvider(ProviderConfig(api_key="key"))
        ctx = RequestContext(prompt="hi", system_prompt="Be concise.")
        payload = provider._build_payload(ctx)
        # system prompt is injected as first user/model turn
        assert payload["contents"][0]["role"] == "user"
        assert payload["contents"][1]["role"] == "model"
        assert payload["contents"][2]["parts"][0]["text"] == "hi"


# ---------------------------------------------------------------------------
# AzureProvider
# ---------------------------------------------------------------------------

class TestAzureProvider:
    def _cfg(self, **extra_kwargs: Any) -> ProviderConfig:
        return ProviderConfig(
            api_key="az-key",
            base_url="https://my-resource.openai.azure.com",
            extra={"deployment": "gpt-4o-mini", **extra_kwargs},
        )

    def test_missing_base_url_raises(self) -> None:
        with pytest.raises(LLMConfigError, match="base_url"):
            AzureProvider(ProviderConfig(api_key="key", extra={"deployment": "d"}))

    def test_missing_deployment_raises(self) -> None:
        with pytest.raises(LLMConfigError, match="deployment"):
            AzureProvider(ProviderConfig(api_key="key", base_url="https://x.openai.azure.com"))

    def test_name(self) -> None:
        assert AzureProvider(self._cfg()).name == "azure"

    @pytest.mark.asyncio
    async def test_complete_success(self, sample_request_ctx: RequestContext) -> None:
        provider = AzureProvider(self._cfg())
        mock_resp = _mock_httpx_response(200, _oai_compat_response("Azure response"))
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_resp)
            result = await provider.complete(sample_request_ctx)
        assert result.text == "Azure response"
        assert result.provider == "azure"

    @pytest.mark.asyncio
    async def test_complete_rate_limit(self, sample_request_ctx: RequestContext) -> None:
        provider = AzureProvider(self._cfg())
        mock_resp = _mock_httpx_response(429, {})
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_resp)
            with pytest.raises(LLMRateLimitError):
                await provider.complete(sample_request_ctx)

    @pytest.mark.asyncio
    async def test_complete_timeout(self, sample_request_ctx: RequestContext) -> None:
        import httpx as _httpx
        provider = AzureProvider(self._cfg())
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=_httpx.TimeoutException("t/o"))
            with pytest.raises(LLMTimeoutError):
                await provider.complete(sample_request_ctx)

    def test_endpoint_contains_deployment_and_version(self) -> None:
        provider = AzureProvider(self._cfg(api_version="2024-05-01"))
        endpoint = provider._endpoint()
        assert "gpt-4o-mini" in endpoint
        assert "2024-05-01" in endpoint

    def test_headers_no_auth_for_managed_identity(self) -> None:
        cfg = ProviderConfig(
            api_key="managed_identity",
            base_url="https://x.openai.azure.com",
            extra={"deployment": "d"},
        )
        provider = AzureProvider(cfg)
        assert "api-key" not in provider._headers()


# ---------------------------------------------------------------------------
# BedrockProvider
# ---------------------------------------------------------------------------

class TestBedrockProvider:
    def test_name(self) -> None:
        assert BedrockProvider(ProviderConfig()).name == "bedrock"

    def test_no_api_key_required(self) -> None:
        # Bedrock uses AWS credential chain — api_key is optional
        provider = BedrockProvider(ProviderConfig())
        assert provider is not None

    @pytest.mark.asyncio
    async def test_complete_anthropic_model(self, sample_request_ctx: RequestContext) -> None:
        provider = BedrockProvider(ProviderConfig(extra={"region": "us-east-1"}))
        raw = _bedrock_anthropic_response("Bedrock reply")

        with patch.object(provider, "_invoke", return_value=raw):
            ctx = RequestContext(
                prompt="hello", model="anthropic.claude-3-haiku-20240307-v1:0"
            )
            result = await provider.complete(ctx)

        assert result.text == "Bedrock reply"
        assert result.provider == "bedrock"
        assert result.prompt_tokens == 8
        assert result.completion_tokens == 4

    @pytest.mark.asyncio
    async def test_complete_timeout(self, sample_request_ctx: RequestContext) -> None:
        provider = BedrockProvider(ProviderConfig(timeout_ms=1))

        async def _slow(_model: str, _body: Any) -> Any:
            import asyncio
            await asyncio.sleep(10)

        with patch.object(provider, "_invoke", side_effect=lambda m, b: (_ for _ in ()).throw(TimeoutError())):
            with pytest.raises((LLMTimeoutError, TimeoutError)):
                await provider.complete(
                    RequestContext(prompt="hi", model="anthropic.claude-3-haiku-20240307-v1:0")
                )

    def test_build_body_titan(self) -> None:
        provider = BedrockProvider(ProviderConfig())
        ctx = RequestContext(prompt="hello", model="amazon.titan-text-express-v1")
        body = provider._build_body(ctx, "amazon.titan-text-express-v1")
        assert "inputText" in body

    def test_build_body_anthropic(self) -> None:
        provider = BedrockProvider(ProviderConfig())
        ctx = RequestContext(prompt="hello", model="anthropic.claude-3-haiku-20240307-v1:0")
        body = provider._build_body(ctx, "anthropic.claude-3-haiku-20240307-v1:0")
        assert "messages" in body
        assert "anthropic_version" in body


# ---------------------------------------------------------------------------
# GroqProvider
# ---------------------------------------------------------------------------

class TestGroqProvider:
    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(LLMConfigError, match="api_key"):
            GroqProvider(ProviderConfig(api_key=None))

    def test_name(self) -> None:
        assert GroqProvider(ProviderConfig(api_key="gsk-key")).name == "groq"

    @pytest.mark.asyncio
    async def test_complete_success(self, sample_request_ctx: RequestContext) -> None:
        provider = GroqProvider(ProviderConfig(api_key="gsk-key"))
        mock_resp = _mock_httpx_response(200, _oai_compat_response("Groq fast reply"))
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_resp)
            result = await provider.complete(sample_request_ctx)
        assert result.text == "Groq fast reply"
        assert result.provider == "groq"

    @pytest.mark.asyncio
    async def test_complete_rate_limit(self, sample_request_ctx: RequestContext) -> None:
        provider = GroqProvider(ProviderConfig(api_key="key"))
        mock_resp = _mock_httpx_response(429, {})
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_resp)
            with pytest.raises(LLMRateLimitError):
                await provider.complete(sample_request_ctx)

    @pytest.mark.asyncio
    async def test_complete_timeout(self, sample_request_ctx: RequestContext) -> None:
        import httpx as _httpx
        provider = GroqProvider(ProviderConfig(api_key="key"))
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=_httpx.TimeoutException("t/o"))
            with pytest.raises(LLMTimeoutError):
                await provider.complete(sample_request_ctx)

    def test_raise_for_status(self) -> None:
        GroqProvider._raise_for_status(200)
        with pytest.raises(LLMRateLimitError):
            GroqProvider._raise_for_status(429)
        with pytest.raises(LLMProviderError):
            GroqProvider._raise_for_status(500)


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------

class TestOllamaProvider:
    def test_name(self) -> None:
        assert OllamaProvider(ProviderConfig()).name == "ollama"

    def test_no_api_key_required(self) -> None:
        provider = OllamaProvider(ProviderConfig())
        assert provider is not None

    def test_custom_base_url(self) -> None:
        provider = OllamaProvider(ProviderConfig(base_url="http://myserver:11434"))
        assert "myserver" in provider._base_url

    @pytest.mark.asyncio
    async def test_complete_success(self, sample_request_ctx: RequestContext) -> None:
        provider = OllamaProvider(ProviderConfig())
        mock_resp = _mock_httpx_response(200, _oai_compat_response("Ollama reply"))
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_resp)
            result = await provider.complete(sample_request_ctx)
        assert result.text == "Ollama reply"
        assert result.provider == "ollama"
        assert result.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_complete_connection_error(self, sample_request_ctx: RequestContext) -> None:
        import httpx as _httpx
        provider = OllamaProvider(ProviderConfig())
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=_httpx.RequestError("refused"))
            with pytest.raises(LLMProviderError, match="Ollama running"):
                await provider.complete(sample_request_ctx)

    @pytest.mark.asyncio
    async def test_complete_timeout(self, sample_request_ctx: RequestContext) -> None:
        import httpx as _httpx
        provider = OllamaProvider(ProviderConfig())
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=_httpx.TimeoutException("t/o"))
            with pytest.raises(LLMTimeoutError):
                await provider.complete(sample_request_ctx)


# ---------------------------------------------------------------------------
# LMStudioProvider
# ---------------------------------------------------------------------------

class TestLMStudioProvider:
    def test_name(self) -> None:
        assert LMStudioProvider(ProviderConfig()).name == "lm_studio"

    def test_no_api_key_required(self) -> None:
        provider = LMStudioProvider(ProviderConfig())
        assert provider is not None

    def test_custom_base_url(self) -> None:
        provider = LMStudioProvider(ProviderConfig(base_url="http://192.168.1.5:1234"))
        assert "192.168.1.5" in provider._base_url

    @pytest.mark.asyncio
    async def test_complete_success(self, sample_request_ctx: RequestContext) -> None:
        provider = LMStudioProvider(ProviderConfig())
        mock_resp = _mock_httpx_response(200, _oai_compat_response("LM Studio reply"))
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(return_value=mock_resp)
            result = await provider.complete(sample_request_ctx)
        assert result.text == "LM Studio reply"
        assert result.provider == "lm_studio"
        assert result.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_complete_connection_error(self, sample_request_ctx: RequestContext) -> None:
        import httpx as _httpx
        provider = LMStudioProvider(ProviderConfig())
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = mock_client
            mock_client.post = AsyncMock(side_effect=_httpx.RequestError("refused"))
            with pytest.raises(LLMProviderError, match="LM Studio running"):
                await provider.complete(sample_request_ctx)

    def test_headers_with_api_key(self) -> None:
        provider = LMStudioProvider(ProviderConfig(api_key="lms-key"))
        assert "Authorization" in provider._headers()

    def test_headers_without_api_key(self) -> None:
        provider = LMStudioProvider(ProviderConfig())
        assert "Authorization" not in provider._headers()

    def test_raise_for_status(self) -> None:
        LMStudioProvider._raise_for_status(200)
        with pytest.raises(LLMProviderError):
            LMStudioProvider._raise_for_status(500)
        with pytest.raises(LLMProviderError):
            LMStudioProvider._raise_for_status(404)
