"""
OpenTelemetry (OTLP) metrics exporter.

Exports collected metrics to an OpenTelemetry Collector or any compatible
OTLP endpoint using the ``opentelemetry-sdk`` and
``opentelemetry-exporter-otlp`` packages.

Requires the ``otlp`` optional dependency:
``pip install llm-std-lib[otlp]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_std_lib.metrics.collector import MetricsSnapshot


def _import_otel() -> tuple[Any, Any, Any]:
    """Return (metrics API module, MeterProvider, PeriodicExportingMetricReader)."""
    try:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry.sdk.metrics import MeterProvider  # type: ignore[attr-defined]
        from opentelemetry.sdk.metrics.export import (  # type: ignore[attr-defined]
            PeriodicExportingMetricReader,
        )

        return otel_metrics, MeterProvider, PeriodicExportingMetricReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "opentelemetry-sdk and opentelemetry-exporter-otlp are required for "
            "OTLPExporter. Install them with: pip install llm-std-lib[otlp]"
        ) from exc


def _import_otlp_exporter() -> Any:
    try:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (  # type: ignore[attr-defined]
            OTLPMetricExporter,
        )

        return OTLPMetricExporter
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "opentelemetry-exporter-otlp is required for OTLPExporter. "
            "Install it with: pip install llm-std-lib[otlp]"
        ) from exc


class OTLPExporter:
    """Exports :class:`~llm_std_lib.metrics.collector.MetricsSnapshot` via OTLP.

    Creates an OpenTelemetry ``MeterProvider`` that pushes metrics to any
    OTLP-compatible backend (Jaeger, Datadog, Grafana Tempo, etc.).

    Usage::

        from llm_std_lib.metrics import MetricsCollector, OTLPExporter

        exporter = OTLPExporter(endpoint="http://localhost:4317")
        collector = MetricsCollector(on_record=exporter.update)

    Args:
        endpoint: OTLP gRPC endpoint (default ``"http://localhost:4317"``).
        service_name: ``service.name`` resource attribute.
        export_interval_ms: How often the SDK pushes metrics (milliseconds).
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4317",
        service_name: str = "llm-std-lib",
        export_interval_ms: int = 30_000,
    ) -> None:
        otel_metrics, MeterProvider, PeriodicExportingMetricReader = _import_otel()
        OTLPMetricExporter = _import_otlp_exporter()

        try:
            from opentelemetry.sdk.resources import Resource  # type: ignore[attr-defined]

            resource = Resource({"service.name": service_name})
        except ImportError:  # pragma: no cover
            resource = None

        otlp_exp = OTLPMetricExporter(endpoint=endpoint)
        reader = PeriodicExportingMetricReader(
            otlp_exp, export_interval_millis=export_interval_ms
        )

        provider_kwargs: dict[str, Any] = {"metric_readers": [reader]}
        if resource is not None:
            provider_kwargs["resource"] = resource

        self._provider = MeterProvider(**provider_kwargs)
        otel_metrics.set_meter_provider(self._provider)

        meter = self._provider.get_meter("llm_std_lib", version="0.6.0")
        self._meter = meter

        # Counters
        self._requests = meter.create_counter(
            "llm.requests_total", description="Total LLM requests dispatched"
        )
        self._success = meter.create_counter(
            "llm.success_total", description="Successful LLM responses"
        )
        self._errors = meter.create_counter(
            "llm.error_total", description="Failed LLM requests"
        )
        self._prompt_tokens = meter.create_counter(
            "llm.prompt_tokens_total", description="Input tokens consumed"
        )
        self._completion_tokens = meter.create_counter(
            "llm.completion_tokens_total", description="Output tokens generated"
        )
        self._total_tokens = meter.create_counter(
            "llm.total_tokens_total", description="Total tokens"
        )
        self._cost_usd = meter.create_counter(
            "llm.cost_usd_total", description="Estimated cost USD", unit="USD"
        )
        self._cache_hits = meter.create_counter(
            "llm.cache_hits_total", description="Semantic cache hits"
        )
        self._cache_misses = meter.create_counter(
            "llm.cache_misses_total", description="Semantic cache misses"
        )
        self._calls_model = meter.create_counter(
            "llm.calls_by_model_total", description="Calls by model"
        )
        self._calls_provider = meter.create_counter(
            "llm.calls_by_provider_total", description="Calls by provider"
        )

        # Gauges (observable)
        self._cache_hit_rate_val: float = 0.0
        self._success_rate_val: float = 1.0
        self._latency_p50_val: float = 0.0
        self._latency_p95_val: float = 0.0
        self._latency_p99_val: float = 0.0

        meter.create_observable_gauge(
            "llm.cache_hit_rate",
            callbacks=[lambda _: [(self._cache_hit_rate_val, {})]],
            description="Cache hit rate",
        )
        meter.create_observable_gauge(
            "llm.success_rate",
            callbacks=[lambda _: [(self._success_rate_val, {})]],
            description="Rolling success rate",
        )
        meter.create_observable_gauge(
            "llm.latency_p50_ms",
            callbacks=[lambda _: [(self._latency_p50_val, {})]],
            description="Latency p50 ms",
            unit="ms",
        )
        meter.create_observable_gauge(
            "llm.latency_p95_ms",
            callbacks=[lambda _: [(self._latency_p95_val, {})]],
            description="Latency p95 ms",
            unit="ms",
        )
        meter.create_observable_gauge(
            "llm.latency_p99_ms",
            callbacks=[lambda _: [(self._latency_p99_val, {})]],
            description="Latency p99 ms",
            unit="ms",
        )

        # Delta tracking for counters (OTel counters require deltas)
        self._prev: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, snapshot: MetricsSnapshot) -> None:
        """Push *snapshot* deltas to the configured OTLP endpoint."""
        self._add_delta("requests_total", snapshot.requests_total, self._requests, {})
        self._add_delta("success_total", snapshot.success_total, self._success, {})
        self._add_delta(
            "prompt_tokens_total", snapshot.prompt_tokens_total, self._prompt_tokens, {}
        )
        self._add_delta(
            "completion_tokens_total",
            snapshot.completion_tokens_total,
            self._completion_tokens,
            {},
        )
        self._add_delta(
            "total_tokens_total", snapshot.total_tokens_total, self._total_tokens, {}
        )
        self._add_delta("cost_usd_total", snapshot.cost_usd_total, self._cost_usd, {})
        self._add_delta("cache_hits_total", snapshot.cache_hits_total, self._cache_hits, {})
        self._add_delta(
            "cache_misses_total", snapshot.cache_misses_total, self._cache_misses, {}
        )

        for err_type, count in snapshot.error_by_type.items():
            self._add_delta(
                f"error_total__{err_type}", count, self._errors, {"error_type": err_type}
            )
        for model, count in snapshot.calls_by_model.items():
            self._add_delta(
                f"calls_by_model__{model}", count, self._calls_model, {"model": model}
            )
        for provider, count in snapshot.calls_by_provider.items():
            self._add_delta(
                f"calls_by_provider__{provider}",
                count,
                self._calls_provider,
                {"provider": provider},
            )

        self._cache_hit_rate_val = snapshot.cache_hit_rate
        self._success_rate_val = snapshot.success_rate
        self._latency_p50_val = snapshot.latency_p50_ms
        self._latency_p95_val = snapshot.latency_p95_ms
        self._latency_p99_val = snapshot.latency_p99_ms

    def shutdown(self) -> None:
        """Flush and shut down the ``MeterProvider``."""
        self._provider.shutdown()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_delta(
        self, key: str, current: float, counter: Any, attrs: dict[str, str]
    ) -> None:
        prev = self._prev.get(key, 0.0)
        delta = current - prev
        if delta > 0:
            counter.add(delta, attrs)
            self._prev[key] = current
