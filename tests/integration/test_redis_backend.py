"""
Integration tests for the Redis cache backend.

Requires a running Redis instance. Start with:
    docker compose up -d redis

Environment variable REDIS_URL controls connection (default: redis://localhost:6379).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


def _skip_if_no_redis() -> None:
    try:
        import redis as _redis  # noqa: F401
    except ImportError:
        pytest.skip("redis package not installed")


@pytest.fixture
async def backend():
    _skip_if_no_redis()
    from llm_std_lib.cache.backends.redis import RedisBackend

    b = RedisBackend(url=REDIS_URL, ttl=60)
    yield b
    await b.clear()


@pytest.mark.asyncio
async def test_store_and_search(backend):
    from llm_std_lib.cache.backends.base import CacheEntry

    vec = np.random.rand(128).astype(np.float32)
    vec /= np.linalg.norm(vec)

    entry = CacheEntry(
        key="integ-1",
        prompt="hello integration",
        response_text="world",
        vector=vec,
        namespace="test",
    )
    await backend.store(entry)

    results = await backend.search(vec, threshold=0.9, namespace="test")
    assert len(results) == 1
    assert results[0].key == "integ-1"
    assert results[0].response_text == "world"


@pytest.mark.asyncio
async def test_delete(backend):
    from llm_std_lib.cache.backends.base import CacheEntry

    vec = np.ones(128, dtype=np.float32)
    vec /= np.linalg.norm(vec)

    entry = CacheEntry(
        key="integ-del",
        prompt="delete me",
        response_text="bye",
        vector=vec,
        namespace="test",
    )
    await backend.store(entry)
    deleted = await backend.delete("integ-del", namespace="test")
    assert deleted is True

    results = await backend.search(vec, threshold=0.9, namespace="test")
    assert all(r.key != "integ-del" for r in results)


@pytest.mark.asyncio
async def test_clear_returns_count(backend):
    from llm_std_lib.cache.backends.base import CacheEntry

    for i in range(3):
        vec = np.random.rand(128).astype(np.float32)
        vec /= np.linalg.norm(vec)
        entry = CacheEntry(
            key=f"integ-clear-{i}",
            prompt=f"prompt {i}",
            response_text=f"resp {i}",
            vector=vec,
            namespace="test",
        )
        await backend.store(entry)

    removed = await backend.clear(namespace="test")
    assert removed >= 3


@pytest.mark.asyncio
async def test_invalidate_by_tag(backend):
    from llm_std_lib.cache.backends.base import CacheEntry

    vec = np.random.rand(128).astype(np.float32)
    vec /= np.linalg.norm(vec)

    entry = CacheEntry(
        key="integ-tag",
        prompt="tagged",
        response_text="tagged response",
        vector=vec,
        namespace="test",
        tags={"integ-tag"},
    )
    await backend.store(entry)

    removed = await backend.invalidate_by_tag("integ-tag", namespace="test")
    assert removed >= 1
