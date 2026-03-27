"""
Latency-optimised routing strategy.

Selects the provider/model with the lowest observed P95 latency from a
rolling history of completed requests. Falls back to the first model in the
candidate list when no historical data is available.
"""

from __future__ import annotations

import math
from collections import deque

from llm_std_lib.exceptions import LLMConfigError
from llm_std_lib.router.strategies.base import BaseStrategy, RouteResult
from llm_std_lib.types import RequestContext

_DEFAULT_WINDOW = 100  # rolling window size for latency samples


class LatencyOptimizedStrategy(BaseStrategy):
    """Routes requests to the model with the lowest historical P95 latency.

    Maintains a sliding window of latency observations per model. When no
    history exists for a model, it is treated as having latency 0 ms (i.e.
    untested models are tried first to gather data).

    Args:
        models: List of ``"provider/model"`` candidates to consider.
        window_size: Number of recent latency samples to keep per model.

    Example::

        strategy = LatencyOptimizedStrategy(
            models=["anthropic/claude-haiku-3", "openai/gpt-4o-mini", "groq/llama-3-8b-8192"],
        )

        # After requests complete, record observations:
        strategy.record("anthropic/claude-haiku-3", latency_ms=180.0)
        strategy.record("openai/gpt-4o-mini", latency_ms=250.0)
    """

    def __init__(self, models: list[str], window_size: int = _DEFAULT_WINDOW) -> None:
        if not models:
            raise LLMConfigError("LatencyOptimizedStrategy requires at least one model.")
        self._models = models
        self._window: dict[str, deque[float]] = {
            m: deque(maxlen=window_size) for m in models
        }

    def route(self, ctx: RequestContext) -> RouteResult:
        """Return the model with the lowest P95 latency."""
        best_model = min(self._models, key=self._p95)
        provider, model = _split_model(best_model)
        p95 = self._p95(best_model)
        reason = (
            f"latency_optimized: P95={p95:.1f}ms"
            if p95 < float("inf")
            else "latency_optimized: no history, using first model"
        )
        return RouteResult(provider=provider, model=model, reason=reason)

    def record(self, full_model: str, latency_ms: float) -> None:
        """Record a latency observation for *full_model*.

        Should be called after each completed request so the strategy can
        update its rolling window.

        Args:
            full_model: ``"provider/model"`` string.
            latency_ms: Observed end-to-end latency in milliseconds.
        """
        if full_model in self._window:
            self._window[full_model].append(latency_ms)

    def _p95(self, full_model: str) -> float:
        """Return the P95 latency for *full_model*, or ``inf`` if no data."""
        samples = self._window.get(full_model)
        if not samples:
            return float("inf")
        sorted_samples = sorted(samples)
        idx = math.ceil(0.95 * len(sorted_samples)) - 1
        return sorted_samples[max(0, idx)]

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Return latency stats (P50, P95) for all models.

        Returns:
            Dict mapping full model name to ``{"p50": ..., "p95": ..., "count": ...}``.
        """
        stats: dict[str, dict[str, float]] = {}
        for model, samples in self._window.items():
            if not samples:
                stats[model] = {"p50": 0.0, "p95": 0.0, "count": 0.0}
                continue
            sorted_s = sorted(samples)
            p50_idx = math.ceil(0.50 * len(sorted_s)) - 1
            p95_idx = math.ceil(0.95 * len(sorted_s)) - 1
            stats[model] = {
                "p50": sorted_s[max(0, p50_idx)],
                "p95": sorted_s[max(0, p95_idx)],
                "count": float(len(sorted_s)),
            }
        return stats


def _split_model(full_model: str) -> tuple[str, str]:
    if "/" in full_model:
        p, m = full_model.split("/", 1)
        return p, m
    return "openai", full_model
