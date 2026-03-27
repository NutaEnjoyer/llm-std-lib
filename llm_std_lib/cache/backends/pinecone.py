"""
Pinecone vector-store cache backend.

Uses Pinecone managed cloud service for vector storage and similarity search.
Recommended for large-scale deployments (>1M entries) where a fully managed
service is preferred.

Requires the ``pinecone`` optional dependency::

    pip install llm-std-lib[pinecone]
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from llm_std_lib.cache.backends.base import BaseCacheBackend, CacheEntry
from llm_std_lib.exceptions import LLMCacheError


class PineconeBackend(BaseCacheBackend):
    """Pinecone-backed semantic cache.

    Uses a single Pinecone index; namespaces map to Pinecone namespaces.

    Args:
        api_key: Pinecone API key.
        index_name: Name of the Pinecone index to use.
        environment: Pinecone environment (e.g. ``us-east-1-aws``).

    Raises:
        LLMCacheError: If ``pinecone-client`` is not installed or the index
            does not exist.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        environment: str = "us-east-1-aws",
    ) -> None:
        self._api_key = api_key
        self._index_name = index_name
        self._environment = environment
        self._index: Any = None

    def _get_index(self) -> Any:
        if self._index is None:
            try:
                from pinecone import Pinecone
            except ImportError as exc:
                raise LLMCacheError(
                    "pinecone-client is not installed. Run: pip install llm-std-lib[pinecone]"
                ) from exc
            pc = Pinecone(api_key=self._api_key)
            self._index = pc.Index(self._index_name)
        return self._index

    async def search(
        self,
        vector: np.ndarray,
        threshold: float,
        namespace: str = "default",
        top_k: int = 1,
    ) -> list[CacheEntry]:
        """Query Pinecone for the nearest neighbours in *namespace*."""
        index = self._get_index()
        now = time.time()

        results = index.query(
            vector=vector.tolist(),
            top_k=top_k * 2,
            namespace=namespace,
            include_metadata=True,
        )

        entries: list[CacheEntry] = []
        for match in results.get("matches", []):
            if match["score"] < threshold:
                continue
            meta = match.get("metadata", {})
            expires_at = meta.get("expires_at")
            if expires_at and float(expires_at) <= now:
                continue
            entry = CacheEntry(
                key=match["id"],
                prompt=meta.get("prompt", ""),
                response_text=meta.get("response_text", ""),
                vector=vector,  # Pinecone doesn't return vectors by default
                metadata=json.loads(meta.get("extra_metadata", "{}")),
                namespace=namespace,
                expires_at=float(expires_at) if expires_at else None,
                tags=set(json.loads(meta.get("tags", "[]"))),
            )
            entries.append(entry)
            if len(entries) >= top_k:
                break
        return entries

    async def store(self, entry: CacheEntry) -> None:
        """Upsert *entry* into the Pinecone index."""
        index = self._get_index()
        index.upsert(
            vectors=[{
                "id": entry.key,
                "values": entry.vector.tolist(),
                "metadata": {
                    "prompt": entry.prompt,
                    "response_text": entry.response_text,
                    "expires_at": str(entry.expires_at or ""),
                    "tags": json.dumps(list(entry.tags)),
                    "extra_metadata": json.dumps(entry.metadata),
                },
            }],
            namespace=entry.namespace,
        )

    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete a single vector by ID."""
        index = self._get_index()
        try:
            index.delete(ids=[key], namespace=namespace)
            return True
        except Exception:
            return False

    async def clear(self, namespace: str | None = None) -> int:
        """Delete all vectors, optionally in *namespace*."""
        index = self._get_index()
        if namespace:
            stats = index.describe_index_stats()
            ns_stats = stats.get("namespaces", {}).get(namespace, {})
            count = int(ns_stats.get("vector_count", 0))
            index.delete(delete_all=True, namespace=namespace)
            return count
        stats = index.describe_index_stats()
        count = int(stats.get("total_vector_count", 0))
        index.delete(delete_all=True)
        return count

    async def invalidate_by_tag(self, tag: str, namespace: str | None = None) -> int:
        """Pinecone does not support server-side tag filtering natively.

        This implementation fetches matching IDs via metadata filter (requires
        Pinecone pods with ``metadata_config``).
        """
        index = self._get_index()
        kwargs: dict[str, Any] = {
            "filter": {"tags": {"$in": [tag]}},
            "include_metadata": False,
        }
        if namespace:
            kwargs["namespace"] = namespace
        results = index.query(vector=[0.0] * 1536, top_k=10_000, **kwargs)
        ids = [m["id"] for m in results.get("matches", [])]
        if not ids:
            return 0
        index.delete(ids=ids, namespace=namespace or "")
        return len(ids)

    async def size(self, namespace: str | None = None) -> int:
        """Return the total vector count from index stats."""
        index = self._get_index()
        stats = index.describe_index_stats()
        if namespace:
            return int(stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0))
        return int(stats.get("total_vector_count", 0))
