"""
ModelRouter — selects the optimal provider/model for each request.

Delegates the routing decision to a pluggable strategy. Supports built-in
strategies (complexity_based, cost_optimized, latency_optimized, round_robin)
and custom Python callables.

Usage::

    from llm_std_lib import ModelRouter
    from llm_std_lib.router.strategies.complexity_based import ComplexityBasedStrategy, Tier

    router = ModelRouter(
        strategy=ComplexityBasedStrategy(tiers=[
            Tier(models=["openai/gpt-4o-mini"], max_complexity=0.3),
            Tier(models=["openai/gpt-4o"],      max_complexity=1.0),
        ])
    )

    # Or use the convenience factory:
    router = ModelRouter.complexity_based(tiers=[...])
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm_std_lib._logging import get_logger
from llm_std_lib.config import PROVIDER_PRICES
from llm_std_lib.exceptions import LLMConfigError
from llm_std_lib.router.strategies.base import BaseStrategy, RouteResult
from llm_std_lib.types import RequestContext

_log = get_logger(__name__)


class CostCalculator:
    """Calculates estimated request cost based on token counts and provider prices.

    Args:
        cost_weights: Multipliers for input and output token prices.
            Default: ``{"input": 1.0, "output": 3.0}``.

    Example::

        calc = CostCalculator()
        usd = calc.estimate("openai", "gpt-4o-mini", prompt_tokens=500, completion_tokens=200)
    """

    def __init__(self, cost_weights: dict[str, float] | None = None) -> None:
        self._weights = cost_weights or {"input": 1.0, "output": 3.0}

    def estimate(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Estimate cost in USD for the given token counts.

        Args:
            provider: Provider name (e.g. ``"openai"``).
            model: Bare model name (e.g. ``"gpt-4o-mini"``).
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD, or ``0.0`` if pricing is unknown.
        """
        prices = PROVIDER_PRICES.get(provider, {}).get(model)
        if not prices:
            return 0.0
        return (
            (prompt_tokens / 1000) * prices["input"] * self._weights["input"]
            + (completion_tokens / 1000) * prices["output"] * self._weights["output"]
        )

    def cheapest(
        self, models: list[str], prompt_tokens: int = 500, completion_tokens: int = 200
    ) -> str:
        """Return the cheapest ``"provider/model"`` string from *models*.

        Args:
            models: List of ``"provider/model"`` strings.
            prompt_tokens: Assumed prompt token count.
            completion_tokens: Assumed completion token count.

        Returns:
            The ``"provider/model"`` string with the lowest estimated cost.
        """
        def _cost(full_model: str) -> float:
            provider, model = (
                full_model.split("/", 1) if "/" in full_model else ("openai", full_model)
            )
            return self.estimate(provider, model, prompt_tokens, completion_tokens)

        return min(models, key=_cost)


class _CustomStrategy(BaseStrategy):
    """Wraps a user-supplied callable as a routing strategy."""

    def __init__(self, fn: Callable[[RequestContext], RouteResult]) -> None:
        self._fn = fn

    def route(self, ctx: RequestContext) -> RouteResult:
        return self._fn(ctx)


class ModelRouter:
    """Routes incoming requests to the most suitable LLM provider/model.

    The routing decision is fully delegated to the configured strategy, making
    it easy to swap strategies without changing application code.

    Args:
        strategy: A :class:`~llm_std_lib.router.strategies.base.BaseStrategy`
            instance, or a callable ``(RequestContext) -> RouteResult``.

    Example::

        router = ModelRouter(strategy=RoundRobinStrategy(
            models=["openai/gpt-4o-mini", "anthropic/claude-haiku-3"]
        ))
        result = router.route(ctx)
        print(result.full_model)  # "openai/gpt-4o-mini"
    """

    def __init__(self, strategy: BaseStrategy | Callable[[RequestContext], RouteResult]) -> None:
        if callable(strategy) and not isinstance(strategy, BaseStrategy):
            self._strategy: BaseStrategy = _CustomStrategy(strategy)
        elif isinstance(strategy, BaseStrategy):
            self._strategy = strategy
        else:
            raise LLMConfigError(
                "strategy must be a BaseStrategy instance or a callable "
                "(RequestContext) -> RouteResult."
            )

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------

    def route(self, ctx: RequestContext) -> RouteResult:
        """Delegate routing to the configured strategy.

        Mutates *ctx* in-place: sets ``ctx.provider`` and ``ctx.model``
        to the selected values.

        Args:
            ctx: The request context to route.

        Returns:
            :class:`RouteResult` with the selected provider and model.
        """
        result = self._strategy.route(ctx)
        ctx.provider = result.provider
        ctx.model = result.model
        _log.debug(
            "router decision",
            provider=result.provider,
            model=result.model,
            reason=result.reason,
            request_id=ctx.request_id,
        )
        return result

    # ------------------------------------------------------------------
    # Convenience factory methods
    # ------------------------------------------------------------------

    @classmethod
    def complexity_based(
        cls,
        tiers: list[dict[str, Any]],
        scorer: object | None = None,
    ) -> ModelRouter:
        """Create a complexity-based router from a list of tier dicts.

        Args:
            tiers: List of dicts with ``models`` and ``max_complexity`` keys.
            scorer: Optional custom :class:`~llm_std_lib.router.complexity.ComplexityScorer`.

        Returns:
            A :class:`ModelRouter` using :class:`ComplexityBasedStrategy`.

        Example::

            router = ModelRouter.complexity_based(tiers=[
                {"models": ["openai/gpt-4o-mini"], "max_complexity": 0.3},
                {"models": ["openai/gpt-4o"],      "max_complexity": 0.7},
                {"models": ["openai/gpt-4-turbo"], "max_complexity": 1.0},
            ])
        """
        from llm_std_lib.router.strategies.complexity_based import ComplexityBasedStrategy, Tier

        tier_objs = [Tier(models=t["models"], max_complexity=t["max_complexity"]) for t in tiers]
        strategy = ComplexityBasedStrategy(tiers=tier_objs, scorer=scorer)  # type: ignore[arg-type]
        return cls(strategy=strategy)

    @classmethod
    def cost_optimized(
        cls,
        models: list[str],
        quality_threshold: float = 0.0,
        cost_weights: dict[str, float] | None = None,
    ) -> ModelRouter:
        """Create a cost-optimised router.

        Args:
            models: Candidate models in ``"provider/model"`` format.
            quality_threshold: Minimum quality rank (0.0–1.0).
            cost_weights: Input/output price multipliers.

        Returns:
            A :class:`ModelRouter` using :class:`CostOptimizedStrategy`.
        """
        from llm_std_lib.router.strategies.cost_optimized import CostOptimizedStrategy

        return cls(strategy=CostOptimizedStrategy(
            models=models,
            quality_threshold=quality_threshold,
            cost_weights=cost_weights,
        ))

    @classmethod
    def latency_optimized(cls, models: list[str], window_size: int = 100) -> ModelRouter:
        """Create a latency-optimised router.

        Args:
            models: Candidate models in ``"provider/model"`` format.
            window_size: Rolling window size for latency samples.

        Returns:
            A :class:`ModelRouter` using :class:`LatencyOptimizedStrategy`.
        """
        from llm_std_lib.router.strategies.latency_optimized import LatencyOptimizedStrategy

        return cls(strategy=LatencyOptimizedStrategy(models=models, window_size=window_size))

    @classmethod
    def round_robin(cls, models: list[str]) -> ModelRouter:
        """Create a round-robin router.

        Args:
            models: Candidate models in ``"provider/model"`` format.

        Returns:
            A :class:`ModelRouter` using :class:`RoundRobinStrategy`.
        """
        from llm_std_lib.router.strategies.round_robin import RoundRobinStrategy

        return cls(strategy=RoundRobinStrategy(models=models))

    @property
    def strategy(self) -> BaseStrategy:
        """The underlying routing strategy."""
        return self._strategy
