"""Unit tests for llm_std_lib.exceptions."""

from __future__ import annotations

import pytest

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


class TestExceptionHierarchy:
    def test_base_is_exception(self) -> None:
        assert issubclass(LLMError, Exception)

    def test_provider_error_is_llm_error(self) -> None:
        assert issubclass(LLMProviderError, LLMError)

    def test_rate_limit_is_provider_error(self) -> None:
        assert issubclass(LLMRateLimitError, LLMProviderError)
        assert issubclass(LLMRateLimitError, LLMError)

    def test_timeout_is_provider_error(self) -> None:
        assert issubclass(LLMTimeoutError, LLMProviderError)
        assert issubclass(LLMTimeoutError, LLMError)

    def test_all_fallbacks_failed_is_llm_error(self) -> None:
        assert issubclass(LLMAllFallbacksFailedError, LLMError)

    def test_circuit_open_is_llm_error(self) -> None:
        assert issubclass(LLMCircuitOpenError, LLMError)

    def test_config_error_is_llm_error(self) -> None:
        assert issubclass(LLMConfigError, LLMError)

    def test_cache_error_is_llm_error(self) -> None:
        assert issubclass(LLMCacheError, LLMError)

    def test_middleware_error_is_llm_error(self) -> None:
        assert issubclass(LLMMiddlewareError, LLMError)

    def test_validation_error_is_llm_error(self) -> None:
        assert issubclass(LLMValidationError, LLMError)


class TestExceptionRaising:
    def test_catch_base_catches_all(self) -> None:
        for exc_class in [
            LLMProviderError, LLMRateLimitError, LLMTimeoutError,
            LLMAllFallbacksFailedError, LLMCircuitOpenError, LLMConfigError,
            LLMCacheError, LLMMiddlewareError, LLMValidationError,
        ]:
            with pytest.raises(LLMError):
                raise exc_class("test message")

    def test_message_preserved(self) -> None:
        try:
            raise LLMConfigError("api_key is missing")
        except LLMConfigError as exc:
            assert "api_key is missing" in str(exc)

    def test_chained_exception(self) -> None:
        original = ValueError("original")
        try:
            raise LLMProviderError("wrapped") from original
        except LLMProviderError as exc:
            assert exc.__cause__ is original
