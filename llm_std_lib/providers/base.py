"""
Abstract base provider interface.

All provider implementations must subclass BaseProvider and implement the
abstract methods, ensuring a consistent contract across every supported backend.
Adding a new provider requires implementing this class (≤50 lines of code).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from llm_std_lib.types import RequestContext, ResponseContext


class BaseProvider(ABC):
    """Abstract base class for all LLM provider adapters.

    Subclasses translate the library's normalised RequestContext into
    provider-specific API calls and map responses back to ResponseContext.

    Example implementation::

        class MyProvider(BaseProvider):
            name = "myprovider"

            async def complete(self, ctx: RequestContext) -> ResponseContext:
                # Call your API here
                ...

            async def stream(self, ctx: RequestContext) -> AsyncGenerator[str, None]:
                # Yield text chunks
                ...
    """

    #: Unique provider identifier. Must match keys in LLMConfig.providers.
    name: str = ""

    @abstractmethod
    async def complete(self, ctx: RequestContext) -> ResponseContext:
        """Send a completion request and return the full response.

        Args:
            ctx: The request context with prompt, model, and parameters.

        Returns:
            ResponseContext with text, token counts, cost, and latency.

        Raises:
            LLMProviderError: On any provider-side error.
            LLMRateLimitError: On HTTP 429 responses.
            LLMTimeoutError: When the request exceeds configured timeout.
        """

    @abstractmethod
    def stream(self, ctx: RequestContext) -> AsyncGenerator[str, None]:
        """Send a streaming completion request and yield text chunks.

        Args:
            ctx: The request context with prompt, model, and parameters.

        Yields:
            Text chunks as they arrive from the provider.

        Raises:
            LLMProviderError: On any provider-side error.
        """

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate request cost in USD based on token counts.

        Falls back to 0.0 if the model is not in the price table.

        Args:
            model: Model name (bare, without provider prefix).
            prompt_tokens: Number of input tokens consumed.
            completion_tokens: Number of output tokens generated.

        Returns:
            Estimated cost in USD.
        """
        from llm_std_lib.config import PROVIDER_PRICES

        prices = PROVIDER_PRICES.get(self.name, {}).get(model)
        if not prices:
            return 0.0
        cost = (prompt_tokens / 1000) * prices["input"]
        cost += (completion_tokens / 1000) * prices["output"]
        return cost
