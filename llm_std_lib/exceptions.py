"""
Exception hierarchy for llm_std_lib.

All library-specific exceptions derive from :class:`LLMError`, which itself
derives from the built-in :class:`Exception`.  Callers can catch the base
class to handle any error raised by the library, or target a specific
sub-class for fine-grained error handling.
"""


class LLMError(Exception):
    """Base exception for all errors raised by llm_std_lib.

    All other library exceptions are sub-classes of this class, so a single
    ``except LLMError`` block is sufficient to catch every library error.
    """


class LLMProviderError(LLMError):
    """Raised when an upstream LLM provider returns an error response.

    This is the base class for provider-specific errors such as rate-limit
    and timeout errors.  It carries the raw provider name and, optionally,
    the HTTP status code returned by the provider's API.
    """


class LLMRateLimitError(LLMProviderError):
    """Raised when the LLM provider responds with a rate-limit error (HTTP 429).

    Callers should back off and retry after the period indicated by the
    provider, or let the built-in retry/resilience layer handle it
    automatically.
    """


class LLMTimeoutError(LLMProviderError):
    """Raised when a request to an LLM provider exceeds the configured timeout.

    This may indicate a slow provider, an oversized request, or transient
    network issues.  The resilience layer can be configured to retry on
    this error.
    """


class LLMAllFallbacksFailedError(LLMError):
    """Raised when every provider in a :class:`FallbackChain` has failed.

    The exception aggregates the individual errors raised by each provider
    attempt so that callers can inspect the root causes.
    """


class LLMCircuitOpenError(LLMError):
    """Raised when a circuit breaker is open and the request is rejected.

    The circuit breaker opens after a configured number of consecutive
    failures and stays open for a cool-down period before allowing
    requests through again.
    """


class LLMConfigError(LLMError):
    """Raised when the library or a provider is misconfigured.

    Examples include missing API keys, invalid model identifiers, or
    conflicting configuration values.
    """


class LLMCacheError(LLMError):
    """Raised when a cache backend operation fails.

    This covers connection errors, serialisation failures, and any other
    issue that prevents the cache layer from reading or writing entries.
    """


class LLMMiddlewareError(LLMError):
    """Raised when a middleware component encounters an unrecoverable error.

    Middleware errors propagate up through the stack and abort the request
    unless the middleware explicitly suppresses them.
    """


class LLMValidationError(LLMError):
    """Raised when request or response data fails validation.

    This is typically triggered by a validator middleware or by Pydantic
    model validation inside the library's type layer.
    """
