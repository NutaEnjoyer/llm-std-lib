"""
MetricsCollector — central sink for LLM observability data.

Implements :class:`~llm_std_lib.middleware.base.BaseMiddleware` so it plugs
directly into a :class:`~llm_std_lib.middleware.stack.MiddlewareStack`.

Tracked metrics (11):
    1.  ``requests_total``          — total requests dispatched
    2.  ``success_total``           — successful responses
    3.  ``error_total``             — failed requests
    4.  ``prompt_tokens_total``     — input tokens consumed
    5.  ``completion_tokens_total`` — output tokens generated
    6.  ``total_tokens_total``      — prompt + completion
    7.  ``cost_usd_total``          — estimated cost in USD
    8.  ``cache_hits_total``        — semantic cache hits
    9.  ``cache_misses_total``      — semantic cache misses
    10. ``cache_hit_rate``          — hits / (hits + misses)
    11. ``success_rate``            — rolling-window success ratio

Latency percentiles (p50 / p95 / p99) and per-model / per-provider
breakdowns are also available via :meth:`snapshot`.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field

from llm_std_lib.middleware.base import BaseMiddleware
from llm_std_lib.types import RequestContext, ResponseContext

_DEFAULT_WINDOW = 200


@dataclass
class MetricsSnapshot:
    """Point-in-time view of all collected metrics."""

    requests_total: int = 0
    success_total: int = 0
    error_total: int = 0
    error_by_type: dict[str, int] = field(default_factory=dict)

    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    total_tokens_total: int = 0

    cost_usd_total: float = 0.0

    cache_hits_total: int = 0
    cache_misses_total: int = 0
    cache_hit_rate: float = 0.0

    success_rate: float = 1.0

    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    calls_by_model: dict[str, int] = field(default_factory=dict)
    calls_by_provider: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return the snapshot as a plain dictionary."""
        return {
            "requests_total": self.requests_total,
            "success_total": self.success_total,
            "error_total": self.error_total,
            "error_by_type": dict(self.error_by_type),
            "prompt_tokens_total": self.prompt_tokens_total,
            "completion_tokens_total": self.completion_tokens_total,
            "total_tokens_total": self.total_tokens_total,
            "cost_usd_total": self.cost_usd_total,
            "cache_hits_total": self.cache_hits_total,
            "cache_misses_total": self.cache_misses_total,
            "cache_hit_rate": self.cache_hit_rate,
            "success_rate": self.success_rate,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "calls_by_model": dict(self.calls_by_model),
            "calls_by_provider": dict(self.calls_by_provider),
        }


def _percentile(sorted_samples: list[float], p: float) -> float:
    if not sorted_samples:
        return 0.0
    idx = math.ceil(p * len(sorted_samples)) - 1
    return sorted_samples[max(0, idx)]


class MetricsCollector(BaseMiddleware):
    """Collects and aggregates LLM call metrics.

    Works as middleware (plug into :class:`~llm_std_lib.middleware.stack.MiddlewareStack`)
    or driven manually via :meth:`record` / :meth:`record_error`.

    Args:
        window_size: Rolling window for success_rate and latency percentiles.
        on_record: Callback invoked after every :meth:`record` with the
            current :class:`MetricsSnapshot`.

    Example::

        collector = MetricsCollector(on_record=lambda s: print(s.success_rate))
        stack = MiddlewareStack([PIIRedactorMiddleware(), collector])
    """

    def __init__(
        self,
        window_size: int = _DEFAULT_WINDOW,
        on_record: Callable[[MetricsSnapshot], None] | None = None,
    ) -> None:
        self._window_size = window_size
        self._callbacks: list[Callable[[MetricsSnapshot], None]] = []
        if on_record:
            self._callbacks.append(on_record)

        self._requests_total = 0
        self._success_total = 0
        self._error_total = 0
        self._error_by_type: dict[str, int] = defaultdict(int)
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost_usd = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        self._success_window: deque[bool] = deque(maxlen=window_size)
        self._latency_window: deque[float] = deque(maxlen=window_size)
        self._calls_by_model: dict[str, int] = defaultdict(int)
        self._calls_by_provider: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def record(self, ctx: RequestContext, response: ResponseContext) -> None:
        """Record a successful request/response pair."""
        self._requests_total += 1
        self._success_total += 1
        self._prompt_tokens += response.prompt_tokens
        self._completion_tokens += response.completion_tokens
        self._cost_usd += response.cost_usd

        if response.cached:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

        if response.latency_ms:
            self._latency_window.append(response.latency_ms)

        self._success_window.append(True)
        self._calls_by_model[f"{response.provider}/{response.model}"] += 1
        self._calls_by_provider[response.provider] += 1
        self._fire_callbacks()

    def record_error(self, ctx: RequestContext, exc: Exception) -> None:
        """Record a failed request."""
        self._requests_total += 1
        self._error_total += 1
        self._error_by_type[type(exc).__name__] += 1
        self._success_window.append(False)
        self._fire_callbacks()

    def add_callback(self, fn: Callable[[MetricsSnapshot], None]) -> None:
        """Register a callback invoked after every recorded event."""
        self._callbacks.append(fn)

    def reset(self) -> None:
        """Reset all counters and windows."""
        self._requests_total = 0
        self._success_total = 0
        self._error_total = 0
        self._error_by_type.clear()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost_usd = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        self._success_window.clear()
        self._latency_window.clear()
        self._calls_by_model.clear()
        self._calls_by_provider.clear()

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> MetricsSnapshot:
        """Return a point-in-time :class:`MetricsSnapshot`."""
        total_cache = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / total_cache if total_cache > 0 else 0.0
        window = list(self._success_window)
        success_rate = sum(window) / len(window) if window else 1.0
        sorted_lat = sorted(self._latency_window)

        return MetricsSnapshot(
            requests_total=self._requests_total,
            success_total=self._success_total,
            error_total=self._error_total,
            error_by_type=dict(self._error_by_type),
            prompt_tokens_total=self._prompt_tokens,
            completion_tokens_total=self._completion_tokens,
            total_tokens_total=self._prompt_tokens + self._completion_tokens,
            cost_usd_total=self._cost_usd,
            cache_hits_total=self._cache_hits,
            cache_misses_total=self._cache_misses,
            cache_hit_rate=cache_hit_rate,
            success_rate=success_rate,
            latency_p50_ms=_percentile(sorted_lat, 0.50),
            latency_p95_ms=_percentile(sorted_lat, 0.95),
            latency_p99_ms=_percentile(sorted_lat, 0.99),
            calls_by_model=dict(self._calls_by_model),
            calls_by_provider=dict(self._calls_by_provider),
        )

    # ------------------------------------------------------------------
    # BaseMiddleware hooks
    # ------------------------------------------------------------------

    async def post_request(
        self, ctx: RequestContext, response: ResponseContext
    ) -> ResponseContext:
        self.record(ctx, response)
        return response

    async def on_error(self, ctx: RequestContext, exc: Exception) -> None:
        self.record_error(ctx, exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fire_callbacks(self) -> None:
        if self._callbacks:
            snap = self.snapshot()
            for fn in self._callbacks:
                fn(snap)
