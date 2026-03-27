"""
Shared Pydantic models for llm_std_lib.

RequestContext flows through the middleware stack and router before reaching
the provider. ResponseContext flows back through middleware after the provider
call. LLMResponse is the normalised object returned to library callers.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class RequestContext(BaseModel):
    """All data associated with a single LLM request.

    Passed through the middleware stack and the router before being handed
    off to the selected provider. Middleware may read and mutate any field.
    """

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    model: str | None = None
    provider: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class ResponseContext(BaseModel):
    """All data returned from a single LLM response.

    Passed back through the middleware stack after the provider call completes,
    allowing middleware to inspect and mutate the response.
    """

    request_id: str
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class LLMResponse(BaseModel):
    """Normalised response object returned to callers of LLMClient.

    Abstracts over provider-specific response formats and exposes a
    consistent interface regardless of which backend served the request.

    Example::

        response = client.complete("Explain semantic caching")
        print(response.text)
        print(f"Cost: ${response.cost_usd:.6f}, tokens: {response.total_tokens}")
    """

    request_id: str
    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_response_context(cls, ctx: ResponseContext) -> LLMResponse:
        """Build an LLMResponse from a ResponseContext."""
        return cls(**ctx.model_dump())
