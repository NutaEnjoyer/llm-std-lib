"""
Prometheus metrics exporter.

Registers and updates ``prometheus_client`` counters, histograms and gauges
for all tracked LLM metrics so they can be scraped by a Prometheus server.

Requires the ``prometheus`` optional dependency:
``pip install llm-std-lib[prometheus]``.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_std_lib.metrics.collector import MetricsSnapshot


def _import_prometheus() -> object:
    try:
        import prometheus_client

        return prometheus_client
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "prometheus-client is required for PrometheusExporter. "
            "Install it with: pip install llm-std-lib[prometheus]"
        ) from exc


class PrometheusExporter:
    """Exports :class:`~llm_std_lib.metrics.collector.MetricsSnapshot` to Prometheus.

    Usage::

        from llm_std_lib.metrics import MetricsCollector, PrometheusExporter

        exporter = PrometheusExporter()
        collector = MetricsCollector(on_record=exporter.update)
        exporter.start_http_server(port=8000)

    Args:
        namespace: Metric name prefix (default ``"llm"``).
        registry: Custom ``prometheus_client.CollectorRegistry``.
            If *None*, uses the default registry.
    """

    def __init__(
        self,
        namespace: str = "llm",
        registry: object | None = None,
    ) -> None:
        self._namespace = namespace
        pc = _import_prometheus()
        reg = registry or pc.REGISTRY  # type: ignore[attr-defined]

        def _counter(name: str, doc: str, labels: list[str] | None = None) -> object:
            kwargs: dict[str, Any] = {
                "name": f"{namespace}_{name}", "documentation": doc, "registry": reg
            }
            if labels:
                kwargs["labelnames"] = labels
            return pc.Counter(**kwargs)  # type: ignore[attr-defined]

        def _gauge(name: str, doc: str, labels: list[str] | None = None) -> object:
            kwargs: dict[str, Any] = {
                "name": f"{namespace}_{name}", "documentation": doc, "registry": reg
            }
            if labels:
                kwargs["labelnames"] = labels
            return pc.Gauge(**kwargs)  # type: ignore[attr-defined]

        def _histogram(name: str, doc: str, buckets: list[float] | None = None) -> object:
            kwargs: dict[str, Any] = {
                "name": f"{namespace}_{name}", "documentation": doc, "registry": reg
            }
            if buckets:
                kwargs["buckets"] = buckets
            return pc.Histogram(**kwargs)  # type: ignore[attr-defined]

        _LAT_BUCKETS = [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]

        self._requests_total = _counter("requests_total", "Total LLM requests dispatched")
        self._success_total = _counter("success_total", "Successful LLM responses")
        self._error_total = _counter("error_total", "Failed LLM requests", ["error_type"])
        self._prompt_tokens = _counter("prompt_tokens_total", "Input tokens consumed")
        self._completion_tokens = _counter("completion_tokens_total", "Output tokens generated")
        self._total_tokens = _counter("total_tokens_total", "Total tokens (prompt + completion)")
        self._cost_usd = _counter("cost_usd_total", "Estimated cost in USD")
        self._cache_hits = _counter("cache_hits_total", "Semantic cache hits")
        self._cache_misses = _counter("cache_misses_total", "Semantic cache misses")
        self._cache_hit_rate = _gauge("cache_hit_rate", "Cache hit rate (hits / (hits + misses))")
        self._success_rate = _gauge("success_rate", "Rolling-window success ratio")
        self._latency_p50 = _gauge("latency_p50_ms", "Latency p50 in milliseconds")
        self._latency_p95 = _gauge("latency_p95_ms", "Latency p95 in milliseconds")
        self._latency_p99 = _gauge("latency_p99_ms", "Latency p99 in milliseconds")
        self._calls_by_model = _counter(
            "calls_by_model_total", "Calls broken down by model", ["model"]
        )
        self._calls_by_provider = _counter(
            "calls_by_provider_total", "Calls broken down by provider", ["provider"]
        )

        # Shadow counters — Prometheus counters are monotonic but snapshots
        # expose cumulative totals, so we track deltas ourselves.
        self._prev: dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, snapshot: MetricsSnapshot) -> None:
        """Push *snapshot* values into Prometheus metrics.

        Call this as a :meth:`~MetricsCollector.add_callback` callback or
        invoke manually after :meth:`~MetricsCollector.snapshot`.
        """
        with self._lock:
            self._sync_counter("requests_total", snapshot.requests_total, self._requests_total)
            self._sync_counter("success_total", snapshot.success_total, self._success_total)
            self._sync_counter(
                "prompt_tokens_total", snapshot.prompt_tokens_total, self._prompt_tokens
            )
            self._sync_counter(
                "completion_tokens_total",
                snapshot.completion_tokens_total,
                self._completion_tokens,
            )
            self._sync_counter(
                "total_tokens_total", snapshot.total_tokens_total, self._total_tokens
            )
            self._sync_counter("cost_usd_total", snapshot.cost_usd_total, self._cost_usd)
            self._sync_counter("cache_hits_total", snapshot.cache_hits_total, self._cache_hits)
            self._sync_counter(
                "cache_misses_total", snapshot.cache_misses_total, self._cache_misses
            )

            # Per-error-type counter
            for err_type, count in snapshot.error_by_type.items():
                key = f"error_total__{err_type}"
                self._sync_labeled_counter(key, count, self._error_total, err_type)

            # Per-model / per-provider counters
            for model, count in snapshot.calls_by_model.items():
                key = f"calls_by_model__{model}"
                self._sync_labeled_counter(key, count, self._calls_by_model, model)

            for provider, count in snapshot.calls_by_provider.items():
                key = f"calls_by_provider__{provider}"
                self._sync_labeled_counter(key, count, self._calls_by_provider, provider)

            # Gauges — always overwrite
            self._cache_hit_rate.set(snapshot.cache_hit_rate)  # type: ignore[attr-defined]
            self._success_rate.set(snapshot.success_rate)  # type: ignore[attr-defined]
            self._latency_p50.set(snapshot.latency_p50_ms)  # type: ignore[attr-defined]
            self._latency_p95.set(snapshot.latency_p95_ms)  # type: ignore[attr-defined]
            self._latency_p99.set(snapshot.latency_p99_ms)  # type: ignore[attr-defined]

    def start_http_server(self, port: int = 8000, addr: str = "") -> None:
        """Start a lightweight HTTP server exposing ``/metrics`` on *port*.

        Delegates to ``prometheus_client.start_http_server``.
        """
        pc = _import_prometheus()
        pc.start_http_server(port, addr)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_counter(self, key: str, current: float, metric: object) -> None:
        prev = self._prev.get(key, 0.0)
        delta = current - prev
        if delta > 0:
            metric.inc(delta)  # type: ignore[attr-defined]
            self._prev[key] = current

    def _sync_labeled_counter(
        self, key: str, current: float, metric: object, label: str
    ) -> None:
        prev = self._prev.get(key, 0.0)
        delta = current - prev
        if delta > 0:
            metric.labels(label).inc(delta)  # type: ignore[attr-defined]
            self._prev[key] = current
