"""Unit tests for LLMClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from llm_std_lib.client import LLMClient
from llm_std_lib.config import LLMConfig, ProviderConfig
from llm_std_lib.exceptions import LLMConfigError
from llm_std_lib.router.model_router import ModelRouter
from llm_std_lib.router.strategies.base import BaseStrategy, RouteResult
from llm_std_lib.types import LLMResponse, RequestContext, ResponseContext


def _fake_response_ctx(request_id: str) -> ResponseContext:
    return ResponseContext(
        request_id=request_id,
        text="Mocked LLM response.",
        model="gpt-4o-mini",
        provider="openai",
        prompt_tokens=8,
        completion_tokens=4,
        total_tokens=12,
        cost_usd=0.0000018,
        latency_ms=100.0,
    )


class TestLLMClientInit:
    def test_init_with_config(self, llm_config: LLMConfig) -> None:
        client = LLMClient(llm_config)
        assert client.config is llm_config

    def test_from_env_raises_without_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        with pytest.raises(LLMConfigError, match="No provider API keys"):
            LLMClient.from_env()

    def test_from_env_with_openai_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-1234567890abcdef")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        client = LLMClient.from_env()
        assert "openai" in client.config.providers

    def test_unknown_provider_raises_on_init(self) -> None:
        with pytest.warns(UserWarning, match="Unknown provider"):
            cfg = LLMConfig(
                default_model="mycloud/model-x",
                providers={"mycloud": ProviderConfig(api_key="key")},
            )
        with pytest.raises(LLMConfigError):
            LLMClient(cfg)


class TestLLMClientComplete:
    @pytest.mark.asyncio
    async def test_acomplete_returns_llm_response(self, llm_config: LLMConfig) -> None:
        client = LLMClient(llm_config)

        async def fake_complete(ctx: RequestContext) -> ResponseContext:
            return _fake_response_ctx(ctx.request_id)

        client._providers["openai"].complete = fake_complete  # type: ignore[method-assign]

        response = await client.acomplete("What is 2+2?")

        assert isinstance(response, LLMResponse)
        assert response.text == "Mocked LLM response."
        assert response.provider == "openai"
        assert response.total_tokens == 12

    def test_complete_sync_returns_llm_response(self, llm_config: LLMConfig) -> None:
        client = LLMClient(llm_config)

        async def fake_complete(ctx: RequestContext) -> ResponseContext:
            return _fake_response_ctx(ctx.request_id)

        client._providers["openai"].complete = fake_complete  # type: ignore[method-assign]

        response = client.complete("What is 2+2?")
        assert isinstance(response, LLMResponse)
        assert response.text == "Mocked LLM response."

    @pytest.mark.asyncio
    async def test_acomplete_with_model_override(self, llm_config: LLMConfig) -> None:
        client = LLMClient(llm_config)
        captured: list[RequestContext] = []

        async def fake_complete(ctx: RequestContext) -> ResponseContext:
            captured.append(ctx)
            return _fake_response_ctx(ctx.request_id)

        client._providers["openai"].complete = fake_complete  # type: ignore[method-assign]

        await client.acomplete("Hello", model="openai/gpt-4o")
        assert captured[0].model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_acomplete_with_system_prompt(self, llm_config: LLMConfig) -> None:
        client = LLMClient(llm_config)
        captured: list[RequestContext] = []

        async def fake_complete(ctx: RequestContext) -> ResponseContext:
            captured.append(ctx)
            return _fake_response_ctx(ctx.request_id)

        client._providers["openai"].complete = fake_complete  # type: ignore[method-assign]

        await client.acomplete("Hello", system_prompt="Be concise.")
        assert captured[0].system_prompt == "Be concise."

    @pytest.mark.asyncio
    async def test_acomplete_unconfigured_provider_raises(self) -> None:
        cfg = LLMConfig(
            default_model="anthropic/claude-haiku-3",
            providers={"openai": ProviderConfig(api_key="sk-test-key-1234567890abcdef")},
        )
        client = LLMClient(cfg)

        with pytest.raises(LLMConfigError, match="anthropic"):
            await client.acomplete("Hello")

    @pytest.mark.asyncio
    async def test_request_ids_are_unique(self, llm_config: LLMConfig) -> None:
        client = LLMClient(llm_config)
        captured: list[str] = []

        async def fake_complete(ctx: RequestContext) -> ResponseContext:
            captured.append(ctx.request_id)
            return _fake_response_ctx(ctx.request_id)

        client._providers["openai"].complete = fake_complete  # type: ignore[method-assign]

        await client.acomplete("First")
        await client.acomplete("Second")
        assert captured[0] != captured[1]

    @pytest.mark.asyncio
    async def test_router_selects_model(self, llm_config: LLMConfig) -> None:
        """Router overrides model when no explicit model is passed."""

        class _AlwaysHaiku(BaseStrategy):
            def route(self, ctx: RequestContext) -> RouteResult:
                return RouteResult(provider="openai", model="gpt-4o-mini", reason="test")

        client = LLMClient(llm_config, router=ModelRouter(strategy=_AlwaysHaiku()))
        captured: list[RequestContext] = []

        async def fake_complete(ctx: RequestContext) -> ResponseContext:
            captured.append(ctx)
            return _fake_response_ctx(ctx.request_id)

        client._providers["openai"].complete = fake_complete  # type: ignore[method-assign]

        await client.acomplete("route me")
        assert captured[0].model == "gpt-4o-mini"
        assert captured[0].provider == "openai"

    @pytest.mark.asyncio
    async def test_explicit_model_bypasses_router(self, llm_config: LLMConfig) -> None:
        """Explicit model= takes precedence over the router."""

        class _AlwaysTurbo(BaseStrategy):
            def route(self, ctx: RequestContext) -> RouteResult:
                return RouteResult(provider="openai", model="gpt-4-turbo", reason="test")

        client = LLMClient(llm_config, router=ModelRouter(strategy=_AlwaysTurbo()))
        captured: list[RequestContext] = []

        async def fake_complete(ctx: RequestContext) -> ResponseContext:
            captured.append(ctx)
            return _fake_response_ctx(ctx.request_id)

        client._providers["openai"].complete = fake_complete  # type: ignore[method-assign]

        await client.acomplete("pin me", model="openai/gpt-4o")
        assert captured[0].model == "gpt-4o"  # explicit wins

    @pytest.mark.asyncio
    async def test_router_property_and_setter(self, llm_config: LLMConfig) -> None:
        client = LLMClient(llm_config)
        assert client.router is None

        class _Dummy(BaseStrategy):
            def route(self, ctx: RequestContext) -> RouteResult:
                return RouteResult(provider="openai", model="gpt-4o-mini", reason="x")

        router = ModelRouter(strategy=_Dummy())
        client.router = router
        assert client.router is router

        client.router = None
        assert client.router is None

    @pytest.mark.asyncio
    async def test_no_router_uses_default_model(self, llm_config: LLMConfig) -> None:
        """Without a router, the config default_model is used."""
        client = LLMClient(llm_config)
        captured: list[RequestContext] = []

        async def fake_complete(ctx: RequestContext) -> ResponseContext:
            captured.append(ctx)
            return _fake_response_ctx(ctx.request_id)

        client._providers["openai"].complete = fake_complete  # type: ignore[method-assign]

        await client.acomplete("hello")
        assert captured[0].model == llm_config.default_model.split("/", 1)[-1]

    @pytest.mark.asyncio
    async def test_tags_forwarded_to_context(self, llm_config: LLMConfig) -> None:
        client = LLMClient(llm_config)
        captured: list[RequestContext] = []

        async def fake_complete(ctx: RequestContext) -> ResponseContext:
            captured.append(ctx)
            return _fake_response_ctx(ctx.request_id)

        client._providers["openai"].complete = fake_complete  # type: ignore[method-assign]

        await client.acomplete("Hello", tags={"user_id": "u42", "feature": "chat"})
        assert captured[0].tags == {"user_id": "u42", "feature": "chat"}
