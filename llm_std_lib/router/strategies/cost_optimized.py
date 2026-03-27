"""
Cost-optimised routing strategy.

Selects the cheapest provider/model from a candidate list, optionally
respecting a quality threshold that excludes models below a minimum tier.
"""

from __future__ import annotations

from llm_std_lib.config import PROVIDER_PRICES
from llm_std_lib.exceptions import LLMConfigError
from llm_std_lib.router.strategies.base import BaseStrategy, RouteResult
from llm_std_lib.types import RequestContext

# Default assumed token ratio for cost estimation (when no historical data)
_DEFAULT_PROMPT_TOKENS = 500
_DEFAULT_COMPLETION_TOKENS = 200


class CostOptimizedStrategy(BaseStrategy):
    """Routes every request to the cheapest model above *quality_threshold*.

    Models are ranked by estimated cost for a synthetic request of
    ``_DEFAULT_PROMPT_TOKENS`` input tokens and ``_DEFAULT_COMPLETION_TOKENS``
    output tokens. The cheapest model that meets the quality requirement wins.

    Args:
        models: List of ``"provider/model"`` candidates to consider.
        quality_threshold: Minimum quality tier index (0 = any, higher = better).
            Models are ranked from cheapest (0) to most expensive (len-1);
            ``quality_threshold`` is the minimum allowed rank index when
            expressed as a fraction (0.0–1.0).
        cost_weights: Dict with ``input`` and ``output`` multipliers applied to
            the base price. Default: ``{"input": 1.0, "output": 3.0}`` (output
            tokens are typically more expensive to generate).

    Example::

        strategy = CostOptimizedStrategy(
            models=["openai/gpt-4o-mini", "openai/gpt-4o", "anthropic/claude-haiku-3"],
            quality_threshold=0.0,
        )
    """

    def __init__(
        self,
        models: list[str],
        quality_threshold: float = 0.0,
        cost_weights: dict[str, float] | None = None,
    ) -> None:
        if not models:
            raise LLMConfigError("CostOptimizedStrategy requires at least one model.")
        if not 0.0 <= quality_threshold <= 1.0:
            raise LLMConfigError(
                f"quality_threshold must be in [0.0, 1.0], got {quality_threshold}."
            )
        self._models = models
        self._quality_threshold = quality_threshold
        self._weights = cost_weights or {"input": 1.0, "output": 3.0}

    def route(self, ctx: RequestContext) -> RouteResult:
        """Return the cheapest qualifying model."""
        ranked = self._rank_by_cost()

        # Apply quality threshold: skip cheapest fraction
        cutoff = int(len(ranked) * self._quality_threshold)
        candidates = ranked[cutoff:] if cutoff < len(ranked) else ranked

        provider, model = _split_model(candidates[0][0])
        cost = candidates[0][1]
        return RouteResult(
            provider=provider,
            model=model,
            reason=f"cost_optimized: estimated_cost={cost:.8f} USD/req",
        )

    def _rank_by_cost(self) -> list[tuple[str, float]]:
        """Return models sorted by estimated cost (cheapest first)."""
        scored: list[tuple[str, float]] = []
        for full_model in self._models:
            provider, model = _split_model(full_model)
            prices = PROVIDER_PRICES.get(provider, {}).get(model)
            if prices:
                cost = (
                    (_DEFAULT_PROMPT_TOKENS / 1000) * prices["input"] * self._weights["input"]
                    + (_DEFAULT_COMPLETION_TOKENS / 1000)
                    * prices["output"]
                    * self._weights["output"]
                )
            else:
                cost = float("inf")  # unknown price → deprioritise
            scored.append((full_model, cost))
        scored.sort(key=lambda t: t[1])
        return scored


def _split_model(full_model: str) -> tuple[str, str]:
    if "/" in full_model:
        p, m = full_model.split("/", 1)
        return p, m
    return "openai", full_model
