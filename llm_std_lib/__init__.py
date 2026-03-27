"""
llm_std_lib — Python library for standardization, optimization and monitoring of LLM API calls.

This package exposes the main public API for interacting with multiple LLM providers
through a unified interface, with built-in caching, routing, resilience and middleware.
"""

from llm_std_lib.cache.semantic_cache import SemanticCache
from llm_std_lib.client import LLMClient
from llm_std_lib.config import LLMConfig
from llm_std_lib.exceptions import (
    LLMAllFallbacksFailedError,
    LLMCacheError,
    LLMCircuitOpenError,
    LLMConfigError,
    LLMError,
    LLMMiddlewareError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMValidationError,
)
from llm_std_lib.middleware.base import BaseMiddleware
from llm_std_lib.middleware.builtins.cost import CostTrackerMiddleware
from llm_std_lib.middleware.builtins.pii import PIIRedactorMiddleware
from llm_std_lib.resilience.fallback import FallbackChain
from llm_std_lib.router.model_router import ModelRouter
from llm_std_lib.types import LLMResponse, RequestContext, ResponseContext

__version__ = "1.0.1"
__all__ = [
    "LLMClient",
    "LLMConfig",
    "RequestContext",
    "ResponseContext",
    "LLMResponse",
    "LLMError",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMAllFallbacksFailedError",
    "LLMCircuitOpenError",
    "LLMConfigError",
    "LLMCacheError",
    "LLMMiddlewareError",
    "LLMValidationError",
    "SemanticCache",
    "ModelRouter",
    "FallbackChain",
    "BaseMiddleware",
    "PIIRedactorMiddleware",
    "CostTrackerMiddleware",
]
