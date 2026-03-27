"""
Round-robin routing strategy.

Distributes requests evenly across all configured provider/model candidates
in a cyclic fashion, regardless of cost or latency.
"""

from __future__ import annotations

import threading

from llm_std_lib.exceptions import LLMConfigError
from llm_std_lib.router.strategies.base import BaseStrategy, RouteResult
from llm_std_lib.types import RequestContext


class RoundRobinStrategy(BaseStrategy):
    """Cycles through candidates sequentially, one per request.

    Thread-safe: uses a ``threading.Lock`` to protect the counter.

    Args:
        models: List of ``"provider/model"`` candidates to cycle through.

    Example::

        strategy = RoundRobinStrategy(models=[
            "openai/gpt-4o-mini",
            "anthropic/claude-haiku-3",
            "groq/llama-3-8b-8192",
        ])
        result = strategy.route(ctx)  # → openai/gpt-4o-mini
        result = strategy.route(ctx)  # → anthropic/claude-haiku-3
        result = strategy.route(ctx)  # → groq/llama-3-8b-8192
        result = strategy.route(ctx)  # → openai/gpt-4o-mini  (wraps around)
    """

    def __init__(self, models: list[str]) -> None:
        if not models:
            raise LLMConfigError("RoundRobinStrategy requires at least one model.")
        self._models = list(models)
        self._index = 0
        self._lock = threading.Lock()

    def route(self, ctx: RequestContext) -> RouteResult:
        """Return the next model in the cycle."""
        with self._lock:
            full_model = self._models[self._index]
            self._index = (self._index + 1) % len(self._models)

        provider, model = _split_model(full_model)
        return RouteResult(
            provider=provider,
            model=model,
            reason=f"round_robin: slot {self._index}/{len(self._models)}",
        )

    @property
    def current_index(self) -> int:
        """Index of the next model to be selected (0-based)."""
        return self._index


def _split_model(full_model: str) -> tuple[str, str]:
    if "/" in full_model:
        p, m = full_model.split("/", 1)
        return p, m
    return "openai", full_model
