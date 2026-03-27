"""Unit tests for the semantic cache layer (encoders, backends, SemanticCache)."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from llm_std_lib.cache.backends.base import CacheEntry
from llm_std_lib.cache.backends.memory import MemoryBackend
from llm_std_lib.cache.encoders.base import BaseEncoder
from llm_std_lib.cache.semantic_cache import SemanticCache
from llm_std_lib.exceptions import LLMCacheError
from llm_std_lib.types import RequestContext, ResponseContext


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _vec(values: list[float]) -> np.ndarray:
    """Build a normalised float32 vector."""
    v = np.array(values, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


class FakeEncoder(BaseEncoder):
    """Deterministic encoder for testing: returns a fixed vector per text."""

    def __init__(self, mapping: dict[str, np.ndarray] | None = None, dim: int = 4) -> None:
        self._mapping = mapping or {}
        self._dim = dim
        self._default = _vec([1.0, 0.0, 0.0, 0.0])

    @property
    def dimension(self) -> int:
        return self._dim

    async def encode(self, text: str) -> np.ndarray:
        return self._mapping.get(text, self._default)


def _make_entry(
    key: str = "k1",
    prompt: str = "hello",
    response: str = "world",
    vector: np.ndarray | None = None,
    namespace: str = "default",
    expires_at: float | None = None,
    tags: set[str] | None = None,
) -> CacheEntry:
    return CacheEntry(
        key=key,
        prompt=prompt,
        response_text=response,
        vector=vector if vector is not None else _vec([1.0, 0.0, 0.0, 0.0]),
        namespace=namespace,
        expires_at=expires_at,
        tags=tags or set(),
    )


# ---------------------------------------------------------------------------
# BaseEncoder helpers
# ---------------------------------------------------------------------------

class TestBaseEncoderHelpers:
    def test_cosine_similarity_identical(self) -> None:
        v = _vec([1.0, 2.0, 3.0])
        assert abs(BaseEncoder.cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self) -> None:
        a = _vec([1.0, 0.0])
        b = _vec([0.0, 1.0])
        assert abs(BaseEncoder.cosine_similarity(a, b)) < 1e-6

    def test_normalise_unit_vector(self) -> None:
        v = np.array([3.0, 4.0], dtype=np.float32)
        n = BaseEncoder.normalise(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-6

    def test_normalise_zero_vector(self) -> None:
        v = np.array([0.0, 0.0], dtype=np.float32)
        n = BaseEncoder.normalise(v)
        assert np.all(n == 0.0)


# ---------------------------------------------------------------------------
# MemoryBackend
# ---------------------------------------------------------------------------

class TestMemoryBackend:
    async def test_store_and_search_hit(self) -> None:
        backend = MemoryBackend()
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        entry = _make_entry(key="e1", vector=vec)
        await backend.store(entry)

        results = await backend.search(vec, threshold=0.9)
        assert len(results) == 1
        assert results[0].key == "e1"

    async def test_search_miss_below_threshold(self) -> None:
        backend = MemoryBackend()
        vec_a = _vec([1.0, 0.0, 0.0, 0.0])
        vec_b = _vec([0.0, 1.0, 0.0, 0.0])
        await backend.store(_make_entry(key="e1", vector=vec_a))

        results = await backend.search(vec_b, threshold=0.9)
        assert results == []

    async def test_namespace_isolation(self) -> None:
        backend = MemoryBackend()
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        await backend.store(_make_entry(key="e1", vector=vec, namespace="ns_a"))
        await backend.store(_make_entry(key="e2", vector=vec, namespace="ns_b"))

        results_a = await backend.search(vec, threshold=0.9, namespace="ns_a")
        results_b = await backend.search(vec, threshold=0.9, namespace="ns_b")

        assert len(results_a) == 1 and results_a[0].key == "e1"
        assert len(results_b) == 1 and results_b[0].key == "e2"

    async def test_expired_entries_skipped(self) -> None:
        backend = MemoryBackend()
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        past = time.time() - 10  # already expired
        await backend.store(_make_entry(key="e1", vector=vec, expires_at=past))

        results = await backend.search(vec, threshold=0.9)
        assert results == []

    async def test_delete(self) -> None:
        backend = MemoryBackend()
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        await backend.store(_make_entry(key="e1", vector=vec))

        removed = await backend.delete("e1")
        assert removed is True
        results = await backend.search(vec, threshold=0.9)
        assert results == []

    async def test_delete_nonexistent(self) -> None:
        backend = MemoryBackend()
        removed = await backend.delete("no-such-key")
        assert removed is False

    async def test_clear_all(self) -> None:
        backend = MemoryBackend()
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        await backend.store(_make_entry(key="e1", vector=vec, namespace="ns_a"))
        await backend.store(_make_entry(key="e2", vector=vec, namespace="ns_b"))

        removed = await backend.clear()
        assert removed == 2
        assert await backend.size() == 0

    async def test_clear_namespace(self) -> None:
        backend = MemoryBackend()
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        await backend.store(_make_entry(key="e1", vector=vec, namespace="ns_a"))
        await backend.store(_make_entry(key="e2", vector=vec, namespace="ns_b"))

        removed = await backend.clear(namespace="ns_a")
        assert removed == 1
        assert await backend.size(namespace="ns_b") == 1

    async def test_invalidate_by_tag(self) -> None:
        backend = MemoryBackend()
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        await backend.store(_make_entry(key="e1", vector=vec, tags={"prod", "v1"}))
        await backend.store(_make_entry(key="e2", vector=_vec([0.5, 0.5, 0.5, 0.5]), tags={"v1"}))
        await backend.store(_make_entry(key="e3", vector=_vec([0.1, 0.9, 0.0, 0.0]), tags={"prod"}))

        removed = await backend.invalidate_by_tag("v1")
        assert removed == 2
        assert await backend.size() == 1

    async def test_size(self) -> None:
        backend = MemoryBackend()
        assert await backend.size() == 0
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        await backend.store(_make_entry(key="e1", vector=vec))
        await backend.store(_make_entry(key="e2", vector=vec))
        assert await backend.size() == 2

    async def test_max_entries_eviction(self) -> None:
        backend = MemoryBackend(max_entries=2)
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        await backend.store(_make_entry(key="e1", vector=vec))
        await backend.store(_make_entry(key="e2", vector=vec))
        await backend.store(_make_entry(key="e3", vector=vec))  # evicts e1

        # Total in-store: 2 (e2, e3)
        assert len(backend._store) == 2
        assert backend._store[0].key == "e2"

    async def test_top_k(self) -> None:
        backend = MemoryBackend()
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        await backend.store(_make_entry(key="e1", vector=vec))
        await backend.store(_make_entry(key="e2", vector=vec))
        await backend.store(_make_entry(key="e3", vector=vec))

        results = await backend.search(vec, threshold=0.9, top_k=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------

class TestSemanticCache:
    def _make_cache(
        self,
        threshold: float = 0.92,
        ttl: int | None = None,
        namespace: str = "default",
        mapping: dict[str, np.ndarray] | None = None,
    ) -> SemanticCache:
        encoder = FakeEncoder(mapping=mapping)
        backend = MemoryBackend()
        return SemanticCache(
            backend=backend,
            encoder=encoder,
            similarity_threshold=threshold,
            ttl=ttl,
            namespace=namespace,
        )

    def _ctx(self, prompt: str = "What is AI?") -> RequestContext:
        return RequestContext(prompt=prompt, model="gpt-4o-mini", provider="openai")

    def _resp(self, ctx: RequestContext, text: str = "AI is cool.") -> ResponseContext:
        return ResponseContext(
            request_id=ctx.request_id,
            text=text,
            model="gpt-4o-mini",
            provider="openai",
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            cost_usd=0.000001,
            latency_ms=200.0,
        )

    async def test_cache_miss_returns_none(self) -> None:
        cache = self._make_cache()
        ctx = self._ctx()
        result = await cache.lookup(ctx)
        assert result is None
        assert cache.misses == 1
        assert cache.hits == 0

    async def test_cache_hit_after_store(self) -> None:
        prompt = "Explain neural networks"
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        cache = self._make_cache(threshold=0.9, mapping={prompt: vec})
        ctx = self._ctx(prompt)
        resp = self._resp(ctx, "Neural networks are...")

        await cache.store(ctx, resp)
        hit = await cache.lookup(ctx)

        assert hit is not None
        assert hit.text == "Neural networks are..."
        assert hit.cached is True
        assert cache.hits == 1

    async def test_hit_rate_calculation(self) -> None:
        prompt = "Hello"
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        cache = self._make_cache(threshold=0.9, mapping={prompt: vec})
        ctx = self._ctx(prompt)

        await cache.lookup(ctx)  # miss
        await cache.store(ctx, self._resp(ctx))
        await cache.lookup(ctx)  # hit

        assert cache.hit_rate == pytest.approx(0.5)

    async def test_invalid_threshold_raises(self) -> None:
        encoder = FakeEncoder()
        backend = MemoryBackend()
        with pytest.raises(LLMCacheError, match="similarity_threshold"):
            SemanticCache(backend=backend, encoder=encoder, similarity_threshold=1.5)

    async def test_clear(self) -> None:
        prompt = "test"
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        cache = self._make_cache(threshold=0.9, mapping={prompt: vec})
        ctx = self._ctx(prompt)
        await cache.store(ctx, self._resp(ctx))

        removed = await cache.clear()
        assert removed == 1
        assert await cache.size() == 0

    async def test_namespace_isolation(self) -> None:
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        prompt = "test"
        cache_a = SemanticCache(
            backend=MemoryBackend(), encoder=FakeEncoder(mapping={prompt: vec}),
            similarity_threshold=0.9, namespace="ns_a",
        )
        cache_b = SemanticCache(
            backend=cache_a.backend, encoder=FakeEncoder(mapping={prompt: vec}),
            similarity_threshold=0.9, namespace="ns_b",
        )
        ctx = self._ctx(prompt)
        await cache_a.store(ctx, self._resp(ctx))

        # ns_b should not see ns_a's entries
        hit_b = await cache_b.lookup(ctx)
        assert hit_b is None

    async def test_ttl_expiry(self) -> None:
        prompt = "expire me"
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        cache = self._make_cache(threshold=0.9, ttl=1, mapping={prompt: vec})
        ctx = self._ctx(prompt)
        await cache.store(ctx, self._resp(ctx))

        # Manually expire by setting expires_at in the past
        entry = cache.backend._store[0]  # type: ignore[attr-defined]
        object.__setattr__(entry, "expires_at", time.time() - 1)

        hit = await cache.lookup(ctx)
        assert hit is None  # expired

    async def test_invalidate_by_tag(self) -> None:
        prompt = "tagged"
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        cache = self._make_cache(threshold=0.9, mapping={prompt: vec})
        ctx = RequestContext(prompt=prompt, tags={"feature": "chat"})
        await cache.store(ctx, self._resp(ctx))

        removed = await cache.invalidate_by_tag("chat")
        assert removed == 1

    async def test_size(self) -> None:
        prompt = "size test"
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        cache = self._make_cache(threshold=0.9, mapping={prompt: vec})
        assert await cache.size() == 0
        ctx = self._ctx(prompt)
        await cache.store(ctx, self._resp(ctx))
        assert await cache.size() == 1

    async def test_metadata_preserved_in_hit(self) -> None:
        prompt = "meta test"
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        cache = self._make_cache(threshold=0.9, mapping={prompt: vec})
        ctx = self._ctx(prompt)
        resp = self._resp(ctx)
        await cache.store(ctx, resp)

        hit = await cache.lookup(ctx)
        assert hit is not None
        assert hit.model == "gpt-4o-mini"
        assert hit.provider == "openai"
        assert hit.total_tokens == 8

    def test_properties(self) -> None:
        cache = self._make_cache(threshold=0.88, ttl=600, namespace="test-ns")
        assert cache.namespace == "test-ns"
        assert cache.similarity_threshold == 0.88
        assert cache.ttl == 600
        assert isinstance(cache.backend, MemoryBackend)
        assert isinstance(cache.encoder, FakeEncoder)


# ---------------------------------------------------------------------------
# E2E: cache miss → store → cache hit
# ---------------------------------------------------------------------------

class TestSemanticCacheE2E:
    async def test_full_cycle(self) -> None:
        """Complete lifecycle: miss → provider call → store → hit."""
        prompt = "What is the capital of France?"
        vec = _vec([1.0, 0.0, 0.0, 0.0])
        encoder = FakeEncoder(mapping={prompt: vec})
        backend = MemoryBackend()
        cache = SemanticCache(backend=backend, encoder=encoder, similarity_threshold=0.95)

        ctx = RequestContext(prompt=prompt, model="gpt-4o-mini", provider="openai")

        # 1. Lookup — miss
        miss = await cache.lookup(ctx)
        assert miss is None

        # 2. Simulate provider response
        provider_response = ResponseContext(
            request_id=ctx.request_id,
            text="Paris",
            model="gpt-4o-mini",
            provider="openai",
            prompt_tokens=10,
            completion_tokens=1,
            total_tokens=11,
            cost_usd=0.0000017,
            latency_ms=350.0,
        )

        # 3. Store
        await cache.store(ctx, provider_response)

        # 4. Lookup — hit
        hit = await cache.lookup(ctx)
        assert hit is not None
        assert hit.text == "Paris"
        assert hit.cached is True
        assert hit.model == "gpt-4o-mini"
        assert hit.total_tokens == 11

        # 5. Stats
        assert cache.hits == 1
        assert cache.misses == 1
        assert cache.hit_rate == pytest.approx(0.5)
        assert await cache.size() == 1
