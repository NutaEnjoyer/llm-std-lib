"""Unit tests for llm_std_lib.types."""

from __future__ import annotations

import pytest

from llm_std_lib.types import LLMResponse, RequestContext, ResponseContext


class TestRequestContext:
    def test_defaults(self) -> None:
        ctx = RequestContext(prompt="Hello")
        assert ctx.prompt == "Hello"
        assert ctx.model is None
        assert ctx.provider is None
        assert ctx.stream is False
        assert ctx.metadata == {}
        assert ctx.tags == {}
        assert len(ctx.request_id) == 36  # UUID4

    def test_request_id_is_unique(self) -> None:
        a = RequestContext(prompt="Hello")
        b = RequestContext(prompt="Hello")
        assert a.request_id != b.request_id

    def test_explicit_request_id(self) -> None:
        ctx = RequestContext(request_id="my-id", prompt="Hello")
        assert ctx.request_id == "my-id"

    def test_all_fields(self) -> None:
        ctx = RequestContext(
            prompt="Tell me a joke",
            model="gpt-4o",
            provider="openai",
            system_prompt="You are funny.",
            temperature=0.9,
            max_tokens=256,
            stream=True,
            tags={"user_id": "u1"},
            metadata={"feature": "chat"},
        )
        assert ctx.model == "gpt-4o"
        assert ctx.provider == "openai"
        assert ctx.system_prompt == "You are funny."
        assert ctx.temperature == 0.9
        assert ctx.max_tokens == 256
        assert ctx.stream is True
        assert ctx.tags == {"user_id": "u1"}
        assert ctx.metadata == {"feature": "chat"}

    def test_immutable_defaults_not_shared(self) -> None:
        a = RequestContext(prompt="a")
        b = RequestContext(prompt="b")
        a.metadata["key"] = "val"
        assert "key" not in b.metadata


class TestResponseContext:
    def test_defaults(self) -> None:
        ctx = ResponseContext(
            request_id="rid",
            text="Hello",
            model="gpt-4o-mini",
            provider="openai",
        )
        assert ctx.prompt_tokens == 0
        assert ctx.completion_tokens == 0
        assert ctx.total_tokens == 0
        assert ctx.cost_usd == 0.0
        assert ctx.latency_ms == 0.0
        assert ctx.cached is False

    def test_all_fields(self) -> None:
        ctx = ResponseContext(
            request_id="rid",
            text="Answer",
            model="gpt-4o",
            provider="openai",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
            latency_ms=500.0,
            cached=True,
        )
        assert ctx.total_tokens == 30
        assert ctx.cached is True


class TestLLMResponse:
    def test_from_response_context(self, sample_response_ctx: ResponseContext) -> None:
        response = LLMResponse.from_response_context(sample_response_ctx)
        assert response.text == sample_response_ctx.text
        assert response.model == sample_response_ctx.model
        assert response.provider == sample_response_ctx.provider
        assert response.total_tokens == sample_response_ctx.total_tokens
        assert response.cost_usd == sample_response_ctx.cost_usd
        assert response.latency_ms == sample_response_ctx.latency_ms
        assert response.cached == sample_response_ctx.cached

    def test_fields_accessible(self) -> None:
        r = LLMResponse(
            request_id="rid",
            text="Hi",
            model="gpt-4o-mini",
            provider="openai",
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            cost_usd=0.0000012,
            latency_ms=120.0,
        )
        assert r.text == "Hi"
        assert r.total_tokens == 8
