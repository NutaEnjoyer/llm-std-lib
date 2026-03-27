"""
In-memory cache backend.

Stores embeddings and cached responses entirely in process memory using a
plain list + numpy cosine similarity. Suitable for development, testing, and
single-process deployments where persistence across restarts is not needed.

No extra dependencies required beyond ``numpy`` (already a core dependency).
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from llm_std_lib.cache.backends.base import BaseCacheBackend, CacheEntry


class MemoryBackend(BaseCacheBackend):
    """Thread-safe in-memory semantic cache backend.

    Uses a ``asyncio.Lock`` to serialise concurrent writes. Searches scan all
    entries linearly — suitable for up to ~10k entries; for larger workloads
    use the Redis or Qdrant backend.

    Args:
        max_entries: Maximum number of entries before the oldest are evicted
            (LRU-style). ``0`` means unlimited.

    Example::

        backend = MemoryBackend(max_entries=10_000)
        cache = SemanticCache(backend=backend, encoder=encoder)
    """

    def __init__(self, max_entries: int = 0) -> None:
        self._max_entries = max_entries
        # List of (entry, insertion_order) — insertion_order used for LRU eviction
        self._store: list[CacheEntry] = []
        self._lock = asyncio.Lock()

    async def search(
        self,
        vector: np.ndarray,
        threshold: float,
        namespace: str = "default",
        top_k: int = 1,
    ) -> list[CacheEntry]:
        """Linear scan over all entries; O(n) in the number of stored vectors.

        Expired entries are silently skipped (lazy expiry).
        """
        now = time.time()
        candidates: list[tuple[float, CacheEntry]] = []

        async with self._lock:
            for entry in self._store:
                if entry.namespace != namespace:
                    continue
                if entry.expires_at is not None and entry.expires_at <= now:
                    continue
                similarity = float(np.dot(vector, entry.vector))
                if similarity >= threshold:
                    candidates.append((similarity, entry))

        candidates.sort(key=lambda t: t[0], reverse=True)
        return [e for _, e in candidates[:top_k]]

    async def store(self, entry: CacheEntry) -> None:
        """Append *entry* to the in-memory store; evict oldest if at capacity."""
        async with self._lock:
            self._store.append(entry)
            if self._max_entries > 0 and len(self._store) > self._max_entries:
                # Evict the oldest entry (index 0)
                self._store.pop(0)

    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Remove the entry with the given *key* from *namespace*."""
        async with self._lock:
            for i, entry in enumerate(self._store):
                if entry.key == key and entry.namespace == namespace:
                    self._store.pop(i)
                    return True
        return False

    async def clear(self, namespace: str | None = None) -> int:
        """Remove all entries (optionally filtered by namespace)."""
        async with self._lock:
            if namespace is None:
                count = len(self._store)
                self._store.clear()
                return count
            before = len(self._store)
            self._store = [e for e in self._store if e.namespace != namespace]
            return before - len(self._store)

    async def invalidate_by_tag(self, tag: str, namespace: str | None = None) -> int:
        """Remove all entries that carry *tag*, optionally in *namespace*."""
        async with self._lock:
            before = len(self._store)
            self._store = [
                e for e in self._store
                if not (
                    tag in e.tags
                    and (namespace is None or e.namespace == namespace)
                )
            ]
            return before - len(self._store)

    async def size(self, namespace: str | None = None) -> int:
        """Return count of live (non-expired) entries."""
        now = time.time()
        async with self._lock:
            return sum(
                1 for e in self._store
                if (namespace is None or e.namespace == namespace)
                and (e.expires_at is None or e.expires_at > now)
            )
