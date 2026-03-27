"""
LLMClient — unified entry point for sending requests to LLM providers.

Orchestrates providers, the middleware stack, the router, the resilience layer
and the semantic cache via a single, consistent interface.

Typical usage::

    import llm_std_lib as llm

    client = llm.LLMClient.from_env()
    response = client.complete("Explain semantic caching")
    print(response.text)
    print(f"Cost: ${response.cost_usd:.6f}, tokens: {response.total_tokens}")
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import structlog.contextvars

from llm_std_lib._logging import configure_logging, get_logger
from llm_std_lib.config import LLMConfig, ProviderConfig
from llm_std_lib.exceptions import LLMConfigError
from llm_std_lib.providers.base import BaseProvider
from llm_std_lib.types import LLMResponse, RequestContext, ResponseContext

# Imported lazily to avoid circular imports at module level
_ModelRouter = None


def _get_model_router_type() -> type:
    global _ModelRouter
    if _ModelRouter is None:
        from llm_std_lib.router.model_router import ModelRouter  # noqa: PLC0415
        _ModelRouter = ModelRouter
    return _ModelRouter

_log = get_logger(__name__)


def _build_provider(provider_name: str, config: ProviderConfig) -> BaseProvider:
    """Instantiate the correct provider adapter for *provider_name*.

    Args:
        provider_name: One of the supported provider identifiers.
        config: Provider-level configuration.

    Returns:
        Concrete BaseProvider instance.

    Raises:
        LLMConfigError: If *provider_name* is unknown.
    """
    if provider_name == "openai":
        from llm_std_lib.providers.openai import OpenAIProvider
        return OpenAIProvider(config)
    if provider_name == "anthropic":
        from llm_std_lib.providers.anthropic import AnthropicProvider
        return AnthropicProvider(config)
    if provider_name == "google":
        from llm_std_lib.providers.google import GoogleProvider
        return GoogleProvider(config)
    if provider_name == "azure":
        from llm_std_lib.providers.azure import AzureProvider
        return AzureProvider(config)
    if provider_name == "bedrock":
        from llm_std_lib.providers.bedrock import BedrockProvider
        return BedrockProvider(config)
    if provider_name == "ollama":
        from llm_std_lib.providers.ollama import OllamaProvider
        return OllamaProvider(config)
    if provider_name == "groq":
        from llm_std_lib.providers.groq import GroqProvider
        return GroqProvider(config)
    if provider_name == "lm_studio":
        from llm_std_lib.providers.lm_studio import LMStudioProvider
        return LMStudioProvider(config)
    raise LLMConfigError(
        f"Provider '{provider_name}' is not supported. "
        "Supported: openai, anthropic, google, azure, bedrock, ollama, groq, lm_studio."
    )


class LLMClient:
    """High-level client for interacting with LLM providers.

    Orchestrates providers, middleware, model router, fallback chain, circuit
    breaker and semantic cache to provide a single, consistent interface for
    all LLM calls.

    The simplest way to create a client is via :meth:`from_env`, which reads
    API keys from environment variables::

        client = LLMClient.from_env()
        response = client.complete("Hello!")

    For full control, pass an :class:`~llm_std_lib.config.LLMConfig`::

        config = LLMConfig(
            default_model="anthropic/claude-haiku-3",
            providers={"anthropic": ProviderConfig(api_key="sk-ant-...")},
        )
        client = LLMClient(config)

    Args:
        config: Top-level library configuration.
        router: Optional :class:`~llm_std_lib.router.model_router.ModelRouter`.
            When set, every request without an explicit ``model=`` override is
            routed automatically (provider *and* model are selected by the
            strategy).  Explicit ``model=`` always takes precedence over the
            router.
    """

    def __init__(self, config: LLMConfig, *, router: Any | None = None) -> None:
        configure_logging(level=config.log_level)
        self._config = config
        self._providers: dict[str, BaseProvider] = self._init_providers()
        self._router: Any | None = router
        _log.info(
            "LLMClient initialised",
            default_model=config.default_model,
            providers=list(self._providers.keys()),
            router=type(router).__name__ if router is not None else None,
        )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> LLMClient:
        """Create an LLMClient from environment variables.

        Reads ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GOOGLE_API_KEY``,
        ``LLM_STD_DEFAULT_MODEL`` and related variables automatically.

        Returns:
            A fully configured LLMClient instance.

        Raises:
            LLMConfigError: If no provider keys are found in the environment.
        """
        config = LLMConfig.from_env()
        if not config.providers:
            raise LLMConfigError(
                "No provider API keys found in environment. "
                "Set at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY."
            )
        return cls(config)

    # ------------------------------------------------------------------
    # Public API — sync wrappers
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tags: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send a completion request synchronously.

        Blocks the current thread until the response arrives. For non-blocking
        usage, prefer :meth:`acomplete`.

        Args:
            prompt: The user prompt to send to the model.
            model: Override the default model (``"provider/model"`` format).
            system_prompt: Optional system/instruction prompt.
            temperature: Sampling temperature (0.0–2.0).
            max_tokens: Maximum tokens in the completion.
            tags: Arbitrary key-value tags for cost tracking / observability.
            metadata: Arbitrary metadata forwarded through the middleware stack.

        Returns:
            :class:`~llm_std_lib.types.LLMResponse` with text, token counts,
            cost, and latency.

        Raises:
            LLMProviderError: On provider-side failures.
            LLMAllFallbacksFailedError: When all fallbacks are exhausted.
        """
        return asyncio.run(
            self.acomplete(
                prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                tags=tags,
                metadata=metadata,
            )
        )

    # ------------------------------------------------------------------
    # Public API — async
    # ------------------------------------------------------------------

    async def acomplete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tags: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send a completion request asynchronously.

        Args:
            prompt: The user prompt to send to the model.
            model: Override the default model (``"provider/model"`` format).
            system_prompt: Optional system/instruction prompt.
            temperature: Sampling temperature (0.0–2.0).
            max_tokens: Maximum tokens in the completion.
            tags: Arbitrary key-value tags for cost tracking / observability.
            metadata: Arbitrary metadata forwarded through the middleware stack.

        Returns:
            :class:`~llm_std_lib.types.LLMResponse` with text, token counts,
            cost, and latency.

        Raises:
            LLMProviderError: On provider-side failures.
            LLMAllFallbacksFailedError: When all fallbacks are exhausted.
        """
        request_id = str(uuid.uuid4())

        # Bind correlation ID to the structlog context for this request.
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            ctx = self._build_context(
                prompt=prompt,
                request_id=request_id,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                tags=tags or {},
                metadata=metadata or {},
            )

            # Apply router only when the caller did NOT pin a specific model.
            if self._router is not None and model is None:
                self._router.route(ctx)  # mutates ctx.provider + ctx.model in-place

            _log.debug("LLM request started", prompt_length=len(prompt), model=ctx.model)

            response_ctx = await self._dispatch(ctx)
            response = LLMResponse.from_response_context(response_ctx)

            _log.info(
                "LLM request completed",
                model=response.model,
                provider=response.provider,
                total_tokens=response.total_tokens,
                cost_usd=round(response.cost_usd, 8),
                latency_ms=round(response.latency_ms, 1),
                cached=response.cached,
            )
            return response
        finally:
            structlog.contextvars.unbind_contextvars("request_id")

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _build_context(
        self,
        *,
        prompt: str,
        request_id: str,
        model: str | None,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
        tags: dict[str, str],
        metadata: dict[str, Any],
    ) -> RequestContext:
        """Resolve model/provider and build a RequestContext."""
        resolved_model = model or self._config.default_model

        # Strip provider prefix for the context model field
        if "/" in resolved_model:
            provider, model_name = resolved_model.split("/", 1)
        else:
            provider = self._config.default_provider
            model_name = resolved_model

        return RequestContext(
            request_id=request_id,
            prompt=prompt,
            model=model_name,
            provider=provider,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            tags=tags,
            metadata=metadata,
        )

    async def _dispatch(self, ctx: RequestContext) -> ResponseContext:
        """Dispatch the request to the provider selected in *ctx*.

        By the time this is called, ``ctx.provider`` and ``ctx.model`` have
        already been resolved (either from the explicit call arguments, the
        router, or the config defaults).

        Args:
            ctx: Fully populated request context.

        Returns:
            ResponseContext from the provider.

        Raises:
            LLMConfigError: If no suitable provider is configured.
            LLMProviderError: On provider-side failure.
        """
        provider_name = ctx.provider or self._config.default_provider
        provider = self._providers.get(provider_name)

        if provider is None:
            raise LLMConfigError(
                f"Provider '{provider_name}' is not configured. "
                f"Configured providers: {list(self._providers.keys())}. "
                "Add it to LLMConfig.providers with an api_key."
            )

        return await provider.complete(ctx)

    # ------------------------------------------------------------------
    # Provider initialisation
    # ------------------------------------------------------------------

    def _init_providers(self) -> dict[str, BaseProvider]:
        """Instantiate all configured providers at startup (fail-fast)."""
        providers: dict[str, BaseProvider] = {}
        for name, provider_cfg in self._config.providers.items():
            try:
                providers[name] = _build_provider(name, provider_cfg)
            except LLMConfigError:
                raise
            except Exception as exc:
                raise LLMConfigError(
                    f"Failed to initialise provider '{name}': {exc}"
                ) from exc
        return providers

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> LLMConfig:
        """The LLMConfig this client was constructed with."""
        return self._config

    @property
    def router(self) -> Any | None:
        """The active :class:`~llm_std_lib.router.model_router.ModelRouter`, or *None*."""
        return self._router

    @router.setter
    def router(self, value: Any | None) -> None:
        """Swap or remove the router at runtime."""
        self._router = value
