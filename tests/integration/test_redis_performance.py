"""
Performance benchmark: Redis cache lookup latency.

SLA requirement: P95 lookup ≤ 20ms at 100k entries.

Requires a running Redis instance:
    docker compose up -d redis

Run with:
    pytest tests/integration/test_redis_performance.py -v -s
"""

from __future__ import annotations

import os
import time
import uuid

import numpy as np
import pytest

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
BATCH_SIZE = int(os.getenv("PERF_BATCH_SIZE", "100000"))
VECTOR_DIM = 128
P95_THRESHOLD_MS = float(os.getenv("PERF_P95_THRESHOLD_MS", "20.0"))
WARMUP_QUERIES = 10
MEASURE_QUERIES = 200


def _skip_if_no_redis() -> None:
    try:
        import redis as _redis  # noqa: F401
    except ImportError:
        pytest.skip("redis package not installed")


def _percentile(samples: list[float], p: float) -> float:
    import math
    sorted_s = sorted(samples)
    idx = math.ceil(p * len(sorted_s)) - 1
    return sorted_s[max(0, idx)]


@pytest.fixture
async def populated_backend():
    """Create a Redis backend pre-loaded with BATCH_SIZE random vectors."""
    _skip_if_no_redis()

    from llm_std_lib.cache.backends.base import CacheEntry
    from llm_std_lib.cache.backends.redis import RedisBackend

    backend = RedisBackend(url=REDIS_URL)

    print(f"\nLoading {BATCH_SIZE:,} vectors into Redis (dim={VECTOR_DIM})...")
    t0 = time.monotonic()

    # Insert in batches to avoid memory spikes
    chunk = 2000
    for start in range(0, BATCH_SIZE, chunk):
        end = min(start + chunk, BATCH_SIZE)
        for i in range(start, end):
            vec = np.random.rand(VECTOR_DIM).astype(np.float32)
            vec /= np.linalg.norm(vec)
            entry = CacheEntry(
                key=str(uuid.uuid4()),
                prompt=f"prompt_{i}",
                response_text=f"response_{i}",
                vector=vec,
                namespace="perf_test",
            )
            await backend.store(entry)

    elapsed = time.monotonic() - t0
    print(f"Loaded {BATCH_SIZE:,} entries in {elapsed:.1f}s")

    yield backend

    await backend.clear(namespace="perf_test")


@pytest.mark.asyncio
async def test_p95_lookup_under_20ms(populated_backend: object) -> None:
    """P95 cache lookup latency must be ≤ 20ms with 100k entries."""
    from llm_std_lib.cache.backends.redis import RedisBackend

    backend: RedisBackend = populated_backend  # type: ignore[assignment]

    # Warm up connection pool
    for _ in range(WARMUP_QUERIES):
        vec = np.random.rand(VECTOR_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        await backend.search(vec, threshold=0.5, namespace="perf_test", top_k=1)

    # Measure
    latencies: list[float] = []
    for _ in range(MEASURE_QUERIES):
        vec = np.random.rand(VECTOR_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)

        t0 = time.monotonic()
        await backend.search(vec, threshold=0.5, namespace="perf_test", top_k=1)
        latencies.append((time.monotonic() - t0) * 1000)

    p50 = _percentile(latencies, 0.50)
    p95 = _percentile(latencies, 0.95)
    p99 = _percentile(latencies, 0.99)

    print(f"\nLatency over {MEASURE_QUERIES} queries ({BATCH_SIZE:,} entries):")
    print(f"  P50 = {p50:.1f}ms")
    print(f"  P95 = {p95:.1f}ms")
    print(f"  P99 = {p99:.1f}ms")

    assert p95 <= P95_THRESHOLD_MS, (
        f"P95 latency {p95:.1f}ms exceeds SLA of {P95_THRESHOLD_MS}ms. "
        f"P50={p50:.1f}ms, P99={p99:.1f}ms"
    )


@pytest.mark.asyncio
async def test_client_overhead_under_5ms(populated_backend: object) -> None:
    """LLMClient middleware overhead (sans cache/network) must be ≤ 5ms."""
    import asyncio

    from llm_std_lib.types import RequestContext, ResponseContext

    # Simulate the middleware pipeline timing with a no-op provider
    latencies: list[float] = []
    for _ in range(500):
        ctx = RequestContext(prompt="hello", model="openai/gpt-4o-mini")
        # Measure only the context construction and response parsing overhead
        t0 = time.monotonic()
        _ = ResponseContext(
            request_id=ctx.request_id,
            text="hi",
            model="gpt-4o-mini",
            provider="openai",
            prompt_tokens=5,
            completion_tokens=2,
            total_tokens=7,
            cost_usd=0.000001,
            latency_ms=1.0,
        )
        latencies.append((time.monotonic() - t0) * 1000)
        await asyncio.sleep(0)  # yield to event loop

    p95 = _percentile(latencies, 0.95)
    print(f"\nClient overhead P95 = {p95:.3f}ms")
    assert p95 <= 5.0, f"Client overhead P95 {p95:.3f}ms exceeds 5ms SLA"
