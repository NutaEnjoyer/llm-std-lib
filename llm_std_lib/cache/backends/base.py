"""
Abstract cache backend interface.

All backend implementations must subclass BaseCacheBackend and implement the
abstract methods. This ensures a consistent contract regardless of the
underlying storage (in-memory, Redis, ChromaDB, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CacheEntry:
    """A single cached item stored in the backend.

    Attributes:
        key: Unique entry identifier (UUID).
        prompt: Original prompt text.
        response_text: Cached LLM response text.
        vector: Embedding vector for the prompt.
        metadata: Arbitrary key-value metadata (model, cost, tags, etc.).
        namespace: Logical partition for cache isolation.
        expires_at: Unix timestamp (seconds) when the entry expires, or
            ``None`` for no expiry.
        tags: Set of tag strings for group-level invalidation.
    """

    key: str
    prompt: str
    response_text: str
    vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    namespace: str = "default"
    expires_at: float | None = None
    tags: set[str] = field(default_factory=set)


class BaseCacheBackend(ABC):
    """Abstract base class for semantic cache storage backends.

    Adding a new backend requires implementing four methods:
    ``search``, ``store``, ``delete``, and ``clear``.
    """

    @abstractmethod
    async def search(
        self,
        vector: np.ndarray,
        threshold: float,
        namespace: str = "default",
        top_k: int = 1,
    ) -> list[CacheEntry]:
        """Find cached entries whose vectors are similar to *vector*.

        Args:
            vector: Query embedding of shape ``(d,)``.
            threshold: Minimum cosine similarity for a match (0.0–1.0).
            namespace: Only search entries in this namespace.
            top_k: Maximum number of results to return.

        Returns:
            List of matching CacheEntry objects, sorted by similarity
            descending. May be empty if no entry exceeds *threshold*.
        """

    @abstractmethod
    async def store(self, entry: CacheEntry) -> None:
        """Persist a new cache entry.

        Implementations must be atomic: either the full entry is written or
        nothing is written (no partial state).

        Args:
            entry: The cache entry to persist.
        """

    @abstractmethod
    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Remove a cache entry by its key.

        Args:
            key: Entry identifier returned by :class:`CacheEntry`.
            namespace: Namespace the entry belongs to.

        Returns:
            ``True`` if the entry existed and was removed, ``False`` otherwise.
        """

    @abstractmethod
    async def clear(self, namespace: str | None = None) -> int:
        """Remove all entries, optionally filtered by namespace.

        Args:
            namespace: If given, only entries in this namespace are removed.
                If ``None``, the entire cache is cleared.

        Returns:
            Number of entries removed.
        """

    @abstractmethod
    async def invalidate_by_tag(self, tag: str, namespace: str | None = None) -> int:
        """Remove all entries that carry *tag*.

        Args:
            tag: Tag string to match.
            namespace: If given, restrict deletion to this namespace.

        Returns:
            Number of entries removed.
        """

    @abstractmethod
    async def size(self, namespace: str | None = None) -> int:
        """Return the number of entries in the cache.

        Args:
            namespace: Count only entries in this namespace. ``None`` for total.

        Returns:
            Entry count.
        """
