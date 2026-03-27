"""
Shared pytest fixtures and configuration for the llm_std_lib test suite.

Fixtures defined here are automatically available to all test modules in
the ``tests/`` directory tree.
"""

from __future__ import annotations

import pytest

from llm_std_lib.config import LLMConfig, ProviderConfig
from llm_std_lib.types import RequestContext, ResponseContext


@pytest.fixture
def openai_provider_config() -> ProviderConfig:
    """ProviderConfig with a fake OpenAI API key."""
    return ProviderConfig(api_key="sk-test-openai-key-1234567890abcdef")


@pytest.fixture
def anthropic_provider_config() -> ProviderConfig:
    """ProviderConfig with a fake Anthropic API key."""
    return ProviderConfig(api_key="sk-ant-test-anthropic-key-1234567890abcdef")


@pytest.fixture
def llm_config(openai_provider_config: ProviderConfig) -> LLMConfig:
    """Minimal LLMConfig with a single OpenAI provider."""
    return LLMConfig(
        default_model="openai/gpt-4o-mini",
        providers={"openai": openai_provider_config},
    )


@pytest.fixture
def sample_request_ctx() -> RequestContext:
    """A basic RequestContext for testing."""
    return RequestContext(
        request_id="test-request-id-001",
        prompt="What is semantic caching?",
        model="gpt-4o-mini",
        provider="openai",
    )


@pytest.fixture
def sample_response_ctx(sample_request_ctx: RequestContext) -> ResponseContext:
    """A basic ResponseContext matching sample_request_ctx."""
    return ResponseContext(
        request_id=sample_request_ctx.request_id,
        text="Semantic caching stores embeddings to find similar past queries.",
        model="gpt-4o-mini",
        provider="openai",
        prompt_tokens=10,
        completion_tokens=12,
        total_tokens=22,
        cost_usd=0.0000033,
        latency_ms=250.0,
    )
