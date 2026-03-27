"""
Metrics sub-package for llm_std_lib.

Collects, aggregates and exports observability data — including latency
histograms, token counts, cost estimates and error rates — to Prometheus
or OpenTelemetry (OTLP) backends.
"""

from .collector import MetricsCollector, MetricsSnapshot
from .otlp import OTLPExporter
from .prometheus import PrometheusExporter

__all__ = [
    "MetricsCollector",
    "MetricsSnapshot",
    "OTLPExporter",
    "PrometheusExporter",
]
