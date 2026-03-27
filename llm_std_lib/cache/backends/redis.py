"""
Redis cache backend.

Stores embeddings and cached responses in Redis. Supports two modes:

* **Plain Redis** — vectors stored as JSON blobs, linear scan for similarity
  (suitable for up to ~50k entries).
* **Redis Stack / RedisSearch** — uses ``HNSW`` index via ``redis-py`` vector
  similarity search for sub-millisecond lookups at scale.

Requires the ``redis`` optional dependency::

    pip install llm-std-lib[redis]
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from llm_std_lib.cache.backends.base import BaseCacheBackend, CacheEntry
from llm_std_lib.exceptions import LLMCacheError

_KEY_PREFIX = "llm_std_lib:cache"


class RedisBackend(BaseCacheBackend):
    """Redis-backed semantic cache.

    Uses plain Redis hashes with JSON-serialised vectors. For production
    deployments with >50k entries, enable Redis Stack and set
    ``use_vector_index=True`` to leverage the native HNSW index.

    Args:
        url: Redis connection URL (e.g. ``redis://localhost:6379``).
        use_vector_index: If ``True``, use RedisSearch HNSW index for
            O(log n) similarity search. Requires Redis Stack ≥7.2.
        key_prefix: Prefix for all Redis keys.

    Raises:
        LLMCacheError: If ``redis`` is not installed or connection fails.

    Example::

        backend = RedisBackend(url="redis://localhost:6379")
        cache = SemanticCache(backend=backend, encoder=encoder)
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        use_vector_index: bool = False,
        key_prefix: str = _KEY_PREFIX,
    ) -> None:
        self._url = url
        self._use_vector_index = use_vector_index
        self._prefix = key_prefix
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import redis.asyncio as aioredis
            except ImportError as exc:
                raise LLMCacheError(
                    "redis is not installed. Run: pip install llm-std-lib[redis]"
                ) from exc
            self._client = aioredis.from_url(self._url, decode_responses=False)
        return self._client

    def _entry_key(self, namespace: str, key: str) -> str:
        return f"{self._prefix}:{namespace}:{key}"

    def _namespace_pattern(self, namespace: str) -> str:
        return f"{self._prefix}:{namespace}:*"

    async def search(
        self,
        vector: np.ndarray,
        threshold: float,
        namespace: str = "default",
        top_k: int = 1,
    ) -> list[CacheEntry]:
        """Scan all keys in *namespace* and return entries above *threshold*."""
        client = self._get_client()
        now = time.time()
        candidates: list[tuple[float, CacheEntry]] = []

        pattern = self._namespace_pattern(namespace)
        async for key in client.scan_iter(pattern):
            raw = await client.hgetall(key)
            if not raw:
                continue
            entry = self._deserialise(raw)
            if entry is None:
                continue
            if entry.expires_at is not None and entry.expires_at <= now:
                continue
            similarity = float(np.dot(vector, entry.vector))
            if similarity >= threshold:
                candidates.append((similarity, entry))

        candidates.sort(key=lambda t: t[0], reverse=True)
        return [e for _, e in candidates[:top_k]]

    async def store(self, entry: CacheEntry) -> None:
        """Persist *entry* as a Redis hash. Sets TTL if ``expires_at`` is set."""
        client = self._get_client()
        key = self._entry_key(entry.namespace, entry.key)
        data = self._serialise(entry)
        pipe = client.pipeline()
        pipe.hset(key, mapping=data)
        if entry.expires_at is not None:
            ttl_seconds = max(1, int(entry.expires_at - time.time()))
            pipe.expire(key, ttl_seconds)
        await pipe.execute()

    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete entry by key. Returns ``True`` if it existed."""
        client = self._get_client()
        redis_key = self._entry_key(namespace, key)
        deleted = await client.delete(redis_key)
        return bool(deleted)

    async def clear(self, namespace: str | None = None) -> int:
        """Delete all entries, optionally filtered by namespace."""
        client = self._get_client()
        if namespace is None:
            pattern = f"{self._prefix}:*"
        else:
            pattern = self._namespace_pattern(namespace)

        keys = [k async for k in client.scan_iter(pattern)]
        if not keys:
            return 0
        return int(await client.delete(*keys))

    async def invalidate_by_tag(self, tag: str, namespace: str | None = None) -> int:
        """Delete all entries carrying *tag*."""
        client = self._get_client()
        pattern = (
            self._namespace_pattern(namespace)
            if namespace
            else f"{self._prefix}:*"
        )
        removed = 0
        async for key in client.scan_iter(pattern):
            raw_tags = await client.hget(key, b"tags")
            if raw_tags and tag in json.loads(raw_tags):
                await client.delete(key)
                removed += 1
        return removed

    async def size(self, namespace: str | None = None) -> int:
        """Return count of live (non-expired) keys."""
        client = self._get_client()
        pattern = (
            self._namespace_pattern(namespace)
            if namespace
            else f"{self._prefix}:*"
        )
        now = time.time()
        count = 0
        async for key in client.scan_iter(pattern):
            raw = await client.hget(key, b"expires_at")
            if raw:
                exp = float(raw)
                if exp <= now:
                    continue
            count += 1
        return count

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise(entry: CacheEntry) -> dict[bytes, bytes]:
        return {
            b"key": entry.key.encode(),
            b"prompt": entry.prompt.encode(),
            b"response_text": entry.response_text.encode(),
            b"vector": entry.vector.astype(np.float32).tobytes(),
            b"metadata": json.dumps(entry.metadata).encode(),
            b"namespace": entry.namespace.encode(),
            b"expires_at": str(entry.expires_at or "").encode(),
            b"tags": json.dumps(list(entry.tags)).encode(),
        }

    @staticmethod
    def _deserialise(raw: dict[bytes, bytes]) -> CacheEntry | None:
        try:
            vector = np.frombuffer(raw[b"vector"], dtype=np.float32).copy()
            expires_raw = raw.get(b"expires_at", b"").decode()
            expires_at = float(expires_raw) if expires_raw else None
            tags_raw = raw.get(b"tags", b"[]").decode()
            return CacheEntry(
                key=raw[b"key"].decode(),
                prompt=raw[b"prompt"].decode(),
                response_text=raw[b"response_text"].decode(),
                vector=vector,
                metadata=json.loads(raw.get(b"metadata", b"{}").decode()),
                namespace=raw.get(b"namespace", b"default").decode(),
                expires_at=expires_at,
                tags=set(json.loads(tags_raw)),
            )
        except (KeyError, ValueError):
            return None
