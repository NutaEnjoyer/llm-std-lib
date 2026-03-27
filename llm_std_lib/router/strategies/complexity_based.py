"""
Complexity-based routing strategy.

Routes requests to different provider/model tiers based on the estimated
complexity of the incoming prompt. Simple prompts go to cheap/fast models;
complex prompts go to flagship models.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_std_lib.exceptions import LLMConfigError
from llm_std_lib.router.complexity import ComplexityScorer
from llm_std_lib.router.strategies.base import BaseStrategy, RouteResult
from llm_std_lib.types import RequestContext


@dataclass
class Tier:
    """A model tier in the complexity-based routing table.

    Attributes:
        models: List of ``"provider/model"`` strings in this tier.
        max_complexity: Upper bound (exclusive) of complexity scores routed
            to this tier. Tiers must be ordered from lowest to highest.
    """

    models: list[str]
    max_complexity: float

    def __post_init__(self) -> None:
        if not self.models:
            raise LLMConfigError("Each routing tier must have at least one model.")
        if not 0.0 <= self.max_complexity <= 1.0:
            raise LLMConfigError(
                f"Tier max_complexity must be in [0.0, 1.0], got {self.max_complexity}."
            )


class ComplexityBasedStrategy(BaseStrategy):
    """Routes prompts to model tiers by estimated complexity.

    Args:
        tiers: Ordered list of :class:`Tier` objects from lowest to highest
            ``max_complexity``. The last tier is used as a fallback for any
            prompt exceeding all lower thresholds.
        scorer: Optional custom :class:`~llm_std_lib.router.complexity.ComplexityScorer`.

    Example::

        strategy = ComplexityBasedStrategy(tiers=[
            Tier(models=["openai/gpt-4o-mini", "anthropic/claude-haiku-3"], max_complexity=0.3),
            Tier(models=["openai/gpt-4o", "anthropic/claude-sonnet-3-5"],   max_complexity=0.7),
            Tier(models=["openai/gpt-4-turbo", "anthropic/claude-opus-3"],  max_complexity=1.0),
        ])
    """

    def __init__(
        self,
        tiers: list[Tier],
        scorer: ComplexityScorer | None = None,
    ) -> None:
        if not tiers:
            raise LLMConfigError("ComplexityBasedStrategy requires at least one tier.")
        # Sort tiers by max_complexity ascending to ensure correct matching
        self._tiers = sorted(tiers, key=lambda t: t.max_complexity)
        self._scorer = scorer or ComplexityScorer()
        self._tier_index = 0  # round-robin within a tier

    def route(self, ctx: RequestContext) -> RouteResult:
        """Score *ctx.prompt* and return the appropriate tier's first model."""
        complexity = self._scorer.score(ctx.prompt)
        selected_tier = self._tiers[-1]  # default: highest tier

        for tier in self._tiers:
            if complexity <= tier.max_complexity:
                selected_tier = tier
                break

        # Pick first model in the tier (round-robin within tier is a v0.4 feature)
        full_model = selected_tier.models[0]
        provider, model = _split_model(full_model)

        return RouteResult(
            provider=provider,
            model=model,
            reason=f"complexity={complexity:.3f} → tier max={selected_tier.max_complexity}",
        )


def _split_model(full_model: str) -> tuple[str, str]:
    """Split ``'provider/model'`` into ``(provider, model)``."""
    if "/" in full_model:
        provider, model = full_model.split("/", 1)
        return provider, model
    return "openai", full_model
