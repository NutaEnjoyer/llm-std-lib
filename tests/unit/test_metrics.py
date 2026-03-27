"""Unit tests for MetricsCollector (v0.6.0)."""

from __future__ import annotations

import pytest

from llm_std_lib.metrics.collector import MetricsCollector, MetricsSnapshot
from llm_std_lib.types import RequestContext, ResponseContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(prompt: str = "hello") -> RequestContext:
    return RequestContext(prompt=prompt)


def _response(
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    cost_usd: float = 0.001,
    cached: bool = False,
    latency_ms: float | None = 100.0,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
) -> ResponseContext:
    return ResponseContext(
        request_id="test",
        text="ok",
        model=model,
        provider=provider,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost_usd=cost_usd,
        cached=cached,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# MetricsSnapshot.to_dict
# ---------------------------------------------------------------------------


class TestMetricsSnapshot:
    def test_to_dict_keys(self) -> None:
        snap = MetricsSnapshot()
        d = snap.to_dict()
        expected_keys = {
            "requests_total",
            "success_total",
            "error_total",
            "error_by_type",
            "prompt_tokens_total",
            "completion_tokens_total",
            "total_tokens_total",
            "cost_usd_total",
            "cache_hits_total",
            "cache_misses_total",
            "cache_hit_rate",
            "success_rate",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "calls_by_model",
            "calls_by_provider",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_defaults(self) -> None:
        snap = MetricsSnapshot()
        d = snap.to_dict()
        assert d["requests_total"] == 0
        assert d["success_rate"] == 1.0
        assert d["cache_hit_rate"] == 0.0

    def test_to_dict_copies_dicts(self) -> None:
        snap = MetricsSnapshot(error_by_type={"ValueError": 1})
        d = snap.to_dict()
        d["error_by_type"]["ValueError"] = 99
        assert snap.error_by_type["ValueError"] == 1  # original unchanged


# ---------------------------------------------------------------------------
# MetricsCollector — record()
# ---------------------------------------------------------------------------


class TestMetricsCollectorRecord:
    def test_requests_total_increments(self) -> None:
        col = MetricsCollector()
        col.record(_ctx(), _response())
        col.record(_ctx(), _response())
        snap = col.snapshot()
        assert snap.requests_total == 2

    def test_success_total_increments(self) -> None:
        col = MetricsCollector()
        col.record(_ctx(), _response())
        assert col.snapshot().success_total == 1

    def test_prompt_tokens_accumulate(self) -> None:
        col = MetricsCollector()
        col.record(_ctx(), _response(prompt_tokens=100))
        col.record(_ctx(), _response(prompt_tokens=200))
        assert col.snapshot().prompt_tokens_total == 300

    def test_completion_tokens_accumulate(self) -> None:
        col = MetricsCollector()
        col.record(_ctx(), _response(completion_tokens=50))
        assert col.snapshot().completion_tokens_total == 50

    def test_total_tokens_is_sum(self) -> None:
        col = MetricsCollector()
        col.record(_ctx(), _response(prompt_tokens=100, completion_tokens=40))
        snap = col.snapshot()
        assert snap.total_tokens_total == 140

    def test_cost_accumulates(self) -> None:
        col = MetricsCollector()
        col.record(_ctx(), _response(cost_usd=0.01))
        col.record(_ctx(), _response(cost_usd=0.02))
        assert col.snapshot().cost_usd_total == pytest.approx(0.03)

    def test_cache_hit_counted(self) -> None:
        col = MetricsCollector()
        col.record(_ctx(), _response(cached=True))
        col.record(_ctx(), _response(cached=False))
        snap = col.snapshot()
        assert snap.cache_hits_total == 1
        assert snap.cache_misses_total == 1

    def test_cache_hit_rate(self) -> None:
        col = MetricsCollector()
        col.record(_ctx(), _response(cached=True))
        col.record(_ctx(), _response(cached=True))
        col.record(_ctx(), _response(cached=False))
        snap = col.snapshot()
        assert snap.cache_hit_rate == pytest.approx(2 / 3)

    def test_calls_by_model(self) -> None:
        col = MetricsCollector()
        col.record(_ctx(), _response(model="gpt-4o-mini", provider="openai"))
        col.record(_ctx(), _response(model="gpt-4o-mini", provider="openai"))
        snap = col.snapshot()
        assert snap.calls_by_model["openai/gpt-4o-mini"] == 2

    def test_calls_by_provider(self) -> None:
        col = MetricsCollector()
        col.record(_ctx(), _response(provider="openai"))
        col.record(_ctx(), _response(provider="anthropic"))
        snap = col.snapshot()
        assert snap.calls_by_provider["openai"] == 1
        assert snap.calls_by_provider["anthropic"] == 1


# ---------------------------------------------------------------------------
# MetricsCollector — record_error()
# ---------------------------------------------------------------------------


class TestMetricsCollectorRecordError:
    def test_error_total_increments(self) -> None:
        col = MetricsCollector()
        col.record_error(_ctx(), ValueError("boom"))
        assert col.snapshot().error_total == 1

    def test_requests_total_also_increments_on_error(self) -> None:
        col = MetricsCollector()
        col.record_error(_ctx(), RuntimeError("x"))
        assert col.snapshot().requests_total == 1

    def test_error_by_type(self) -> None:
        col = MetricsCollector()
        col.record_error(_ctx(), ValueError("a"))
        col.record_error(_ctx(), ValueError("b"))
        col.record_error(_ctx(), RuntimeError("c"))
        snap = col.snapshot()
        assert snap.error_by_type["ValueError"] == 2
        assert snap.error_by_type["RuntimeError"] == 1

    def test_error_not_counted_in_success(self) -> None:
        col = MetricsCollector()
        col.record(_ctx(), _response())
        col.record_error(_ctx(), Exception("fail"))
        snap = col.snapshot()
        assert snap.success_total == 1
        assert snap.error_total == 1


# ---------------------------------------------------------------------------
# MetricsCollector — success_rate (rolling window)
# ---------------------------------------------------------------------------


class TestSuccessRate:
    def test_default_success_rate_is_one(self) -> None:
        col = MetricsCollector()
        assert col.snapshot().success_rate == 1.0

    def test_success_rate_all_success(self) -> None:
        col = MetricsCollector()
        for _ in range(5):
            col.record(_ctx(), _response())
        assert col.snapshot().success_rate == 1.0

    def test_success_rate_mixed(self) -> None:
        col = MetricsCollector()
        col.record(_ctx(), _response())  # True
        col.record_error(_ctx(), Exception())  # False
        snap = col.snapshot()
        assert snap.success_rate == pytest.approx(0.5)

    def test_success_rate_rolling_window(self) -> None:
        """Window of size 4: after 4 successes + 2 errors, oldest 2 successes drop out."""
        col = MetricsCollector(window_size=4)
        col.record(_ctx(), _response())
        col.record(_ctx(), _response())
        col.record(_ctx(), _response())
        col.record(_ctx(), _response())
        col.record_error(_ctx(), Exception())
        col.record_error(_ctx(), Exception())
        # Window now: [True, True, False, False]
        snap = col.snapshot()
        assert snap.success_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# MetricsCollector — latency percentiles
# ---------------------------------------------------------------------------


class TestLatencyPercentiles:
    def test_no_latency_all_zero(self) -> None:
        col = MetricsCollector()
        # latency_ms=0.0 means "not recorded" — collector skips zero values
        col.record(_ctx(), _response(latency_ms=0.0))
        snap = col.snapshot()
        assert snap.latency_p50_ms == 0.0
        assert snap.latency_p95_ms == 0.0
        assert snap.latency_p99_ms == 0.0

    def test_single_sample(self) -> None:
        col = MetricsCollector()
        col.record(_ctx(), _response(latency_ms=200.0))
        snap = col.snapshot()
        assert snap.latency_p50_ms == 200.0
        assert snap.latency_p95_ms == 200.0
        assert snap.latency_p99_ms == 200.0

    def test_percentiles_ordered(self) -> None:
        col = MetricsCollector()
        for ms in [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0]:
            col.record(_ctx(), _response(latency_ms=ms))
        snap = col.snapshot()
        assert snap.latency_p50_ms <= snap.latency_p95_ms <= snap.latency_p99_ms

    def test_p99_high_outlier(self) -> None:
        # With 51 samples (50 × 10ms + 1 × 9999ms):
        # ceil(0.99 * 51) - 1 = ceil(50.49) - 1 = 51 - 1 = 50 → sorted[50] = 9999.0
        col = MetricsCollector()
        for _ in range(50):
            col.record(_ctx(), _response(latency_ms=10.0))
        col.record(_ctx(), _response(latency_ms=9999.0))
        snap = col.snapshot()
        assert snap.latency_p99_ms == pytest.approx(9999.0)


# ---------------------------------------------------------------------------
# MetricsCollector — reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_counters(self) -> None:
        col = MetricsCollector()
        col.record(_ctx(), _response(cost_usd=0.5, prompt_tokens=100))
        col.record_error(_ctx(), ValueError())
        col.reset()
        snap = col.snapshot()
        assert snap.requests_total == 0
        assert snap.success_total == 0
        assert snap.error_total == 0
        assert snap.prompt_tokens_total == 0
        assert snap.cost_usd_total == 0.0
        assert snap.cache_hits_total == 0
        assert snap.cache_misses_total == 0
        assert snap.success_rate == 1.0
        assert snap.latency_p50_ms == 0.0
        assert snap.error_by_type == {}
        assert snap.calls_by_model == {}
        assert snap.calls_by_provider == {}


# ---------------------------------------------------------------------------
# MetricsCollector — callbacks
# ---------------------------------------------------------------------------


class TestCallbacks:
    def test_on_record_callback_called(self) -> None:
        snaps: list[MetricsSnapshot] = []
        col = MetricsCollector(on_record=snaps.append)
        col.record(_ctx(), _response())
        assert len(snaps) == 1
        assert snaps[0].success_total == 1

    def test_on_error_fires_callback(self) -> None:
        snaps: list[MetricsSnapshot] = []
        col = MetricsCollector(on_record=snaps.append)
        col.record_error(_ctx(), Exception())
        assert len(snaps) == 1
        assert snaps[0].error_total == 1

    def test_add_callback(self) -> None:
        snaps: list[MetricsSnapshot] = []
        col = MetricsCollector()
        col.add_callback(snaps.append)
        col.record(_ctx(), _response())
        assert len(snaps) == 1

    def test_multiple_callbacks(self) -> None:
        a: list[int] = []
        b: list[int] = []
        col = MetricsCollector(
            on_record=lambda s: a.append(s.requests_total)
        )
        col.add_callback(lambda s: b.append(s.success_total))
        col.record(_ctx(), _response())
        assert a == [1]
        assert b == [1]


# ---------------------------------------------------------------------------
# MetricsCollector as BaseMiddleware
# ---------------------------------------------------------------------------


class TestMetricsCollectorMiddleware:
    async def test_post_request_calls_record(self) -> None:
        col = MetricsCollector()
        ctx = _ctx()
        resp = _response()
        result = await col.post_request(ctx, resp)
        assert result is resp
        assert col.snapshot().success_total == 1

    async def test_on_error_calls_record_error(self) -> None:
        col = MetricsCollector()
        await col.on_error(_ctx(), RuntimeError("boom"))
        assert col.snapshot().error_total == 1

    async def test_plugs_into_middleware_stack(self) -> None:
        from llm_std_lib.middleware import MiddlewareStack

        col = MetricsCollector()
        stack = MiddlewareStack([col])

        async def handler(ctx: RequestContext) -> ResponseContext:
            return _response()

        await stack.execute(_ctx(), handler)
        assert col.snapshot().requests_total == 1
        assert col.snapshot().success_total == 1

    async def test_stack_error_recorded(self) -> None:
        from llm_std_lib.middleware import MiddlewareStack

        col = MetricsCollector()
        stack = MiddlewareStack([col])

        async def failing(ctx: RequestContext) -> ResponseContext:
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            await stack.execute(_ctx(), failing)

        snap = col.snapshot()
        assert snap.error_total == 1
        assert snap.requests_total == 1


# ---------------------------------------------------------------------------
# PrometheusExporter — import-error path (no prometheus_client installed)
# ---------------------------------------------------------------------------


class TestPrometheusExporterImportError:
    def test_raises_import_error_when_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import unittest.mock as mock

        # Simulate prometheus_client not installed
        with mock.patch.dict(sys.modules, {"prometheus_client": None}):
            from llm_std_lib.metrics.prometheus import _import_prometheus

            with pytest.raises((ImportError, TypeError)):
                _import_prometheus()


# ---------------------------------------------------------------------------
# OTLPExporter — import-error path (no opentelemetry installed)
# ---------------------------------------------------------------------------


class TestOTLPExporterImportError:
    def test_raises_import_error_when_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import unittest.mock as mock

        otel_mods = [
            "opentelemetry",
            "opentelemetry.metrics",
            "opentelemetry.sdk",
            "opentelemetry.sdk.metrics",
            "opentelemetry.sdk.metrics.export",
        ]
        with mock.patch.dict(sys.modules, {m: None for m in otel_mods}):
            from llm_std_lib.metrics.otlp import _import_otel

            with pytest.raises((ImportError, TypeError)):
                _import_otel()


# ---------------------------------------------------------------------------
# PrometheusExporter — scrape format test
# ---------------------------------------------------------------------------


class TestPrometheusExporterScrapeFormat:
    """Verify that PrometheusExporter registers metrics and update() populates them."""

    def test_update_populates_counters(self) -> None:
        """Counter values are incremented from a snapshot."""
        pc = pytest.importorskip("prometheus_client")
        from prometheus_client import CollectorRegistry

        from llm_std_lib.metrics.collector import MetricsCollector, MetricsSnapshot
        from llm_std_lib.metrics.prometheus import PrometheusExporter

        registry = CollectorRegistry()
        exporter = PrometheusExporter(namespace="test_scrape", registry=registry)

        snapshot = MetricsSnapshot(
            requests_total=10,
            success_total=9,
            error_total=1,
            prompt_tokens_total=500,
            completion_tokens_total=200,
            total_tokens_total=700,
            cost_usd_total=0.005,
            cache_hits_total=3,
            cache_misses_total=7,
            cache_hit_rate=0.3,
            success_rate=0.9,
            latency_p50_ms=120.0,
            latency_p95_ms=340.0,
            latency_p99_ms=500.0,
            calls_by_model={"gpt-4o-mini": 10},
            calls_by_provider={"openai": 10},
            error_by_type={"LLMProviderError": 1},
        )
        exporter.update(snapshot)

        # Collect all metrics from the custom registry
        output = pc.generate_latest(registry).decode()  # type: ignore[attr-defined]
        assert "test_scrape_requests_total" in output
        assert "test_scrape_cache_hit_rate" in output
        assert "test_scrape_latency_p95_ms" in output

    def test_update_is_idempotent_for_unchanged_values(self) -> None:
        """Calling update() twice with the same snapshot does not double-count."""
        pytest.importorskip("prometheus_client")
        from prometheus_client import CollectorRegistry

        from llm_std_lib.metrics.collector import MetricsSnapshot
        from llm_std_lib.metrics.prometheus import PrometheusExporter

        registry = CollectorRegistry()
        exporter = PrometheusExporter(namespace="test_idempotent", registry=registry)

        snapshot = MetricsSnapshot(requests_total=5)
        exporter.update(snapshot)
        exporter.update(snapshot)  # second call — no new delta

        # _prev should hold 5.0, not 10.0
        assert exporter._prev.get("requests_total") == 5.0
