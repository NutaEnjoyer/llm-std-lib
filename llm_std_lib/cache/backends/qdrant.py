"""
Qdrant vector-store cache backend.

Uses Qdrant for persistence and similarity search, with support for payload
filtering (by namespace and expiry).

Requires the ``qdrant`` optional dependency::

    pip install llm-std-lib[qdrant]
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from llm_std_lib.cache.backends.base import BaseCacheBackend, CacheEntry
from llm_std_lib.exceptions import LLMCacheError

_COLLECTION = "llm_std_lib_cache"


class QdrantBackend(BaseCacheBackend):
    """Qdrant-backed semantic cache.

    A single Qdrant collection is used for all namespaces; namespace filtering
    is implemented via payload filters on search and delete.

    Args:
        url: Qdrant server URL (default: ``http://localhost:6333``).
        collection: Qdrant collection name.
        vector_size: Dimensionality of stored vectors (must match encoder).
            Defaults to 1536 (OpenAI ``text-embedding-3-small``).

    Raises:
        LLMCacheError: If ``qdrant-client`` is not installed.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = _COLLECTION,
        vector_size: int = 1536,
    ) -> None:
        self._url = url
        self._collection = collection
        self._vector_size = vector_size
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams
            except ImportError as exc:
                raise LLMCacheError(
                    "qdrant-client is not installed. Run: pip install llm-std-lib[qdrant]"
                ) from exc
            self._client = QdrantClient(url=self._url)
            # Create collection if it does not exist
            existing = [c.name for c in self._client.get_collections().collections]
            if self._collection not in existing:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=self._vector_size, distance=Distance.COSINE
                    ),
                )
        return self._client

    async def search(
        self,
        vector: np.ndarray,
        threshold: float,
        namespace: str = "default",
        top_k: int = 1,
    ) -> list[CacheEntry]:
        """Search with payload filter on namespace and expiry."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = self._get_client()
        now = time.time()

        results = client.search(
            collection_name=self._collection,
            query_vector=vector.tolist(),
            query_filter=Filter(
                must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))]
            ),
            limit=top_k * 2,
            with_payload=True,
            score_threshold=threshold,
        )

        entries: list[CacheEntry] = []
        for hit in results:
            payload = hit.payload or {}
            expires_at = payload.get("expires_at")
            if expires_at and float(expires_at) <= now:
                continue
            vec_raw = payload.get("vector")
            vec = np.array(vec_raw, dtype=np.float32) if vec_raw else vector
            entry = CacheEntry(
                key=str(hit.id),
                prompt=payload.get("prompt", ""),
                response_text=payload.get("response_text", ""),
                vector=vec,
                metadata=json.loads(payload.get("metadata", "{}")),
                namespace=namespace,
                expires_at=float(expires_at) if expires_at else None,
                tags=set(json.loads(payload.get("tags", "[]"))),
            )
            entries.append(entry)
            if len(entries) >= top_k:
                break
        return entries

    async def store(self, entry: CacheEntry) -> None:
        """Upsert entry into the Qdrant collection."""
        from qdrant_client.models import PointStruct

        client = self._get_client()
        client.upsert(
            collection_name=self._collection,
            points=[PointStruct(
                id=entry.key,
                vector=entry.vector.tolist(),
                payload={
                    "prompt": entry.prompt,
                    "response_text": entry.response_text,
                    "namespace": entry.namespace,
                    "expires_at": str(entry.expires_at or ""),
                    "tags": json.dumps(list(entry.tags)),
                    "metadata": json.dumps(entry.metadata),
                    "vector": entry.vector.tolist(),
                },
            )],
        )

    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete a single point by ID."""
        from qdrant_client.models import PointIdsList

        client = self._get_client()
        try:
            client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=[key]),
            )
            return True
        except Exception:
            return False

    async def clear(self, namespace: str | None = None) -> int:
        """Delete all points, optionally filtered by namespace."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = self._get_client()
        if namespace is None:
            count = int(client.count(self._collection).count or 0)
            client.delete_collection(self._collection)
            self._client = None  # Force re-creation next call
            return count

        ns_filter = Filter(
            must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))]
        )
        count = int(client.count(self._collection, count_filter=ns_filter).count or 0)
        client.delete(
            collection_name=self._collection,
            points_selector=ns_filter,
        )
        return count

    async def invalidate_by_tag(self, tag: str, namespace: str | None = None) -> int:
        """Delete all entries carrying *tag*."""
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        client = self._get_client()
        must: list[Any] = [FieldCondition(key="tags", match=MatchAny(any=[tag]))]
        if namespace:
            from qdrant_client.models import MatchValue
            must.append(FieldCondition(key="namespace", match=MatchValue(value=namespace)))

        tag_filter = Filter(must=must)
        count = int(client.count(self._collection, count_filter=tag_filter).count or 0)
        client.delete(collection_name=self._collection, points_selector=tag_filter)
        return count

    async def size(self, namespace: str | None = None) -> int:
        """Return the number of stored points."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = self._get_client()
        if namespace is None:
            return int(client.count(self._collection).count or 0)
        return int(client.count(
            self._collection,
            count_filter=Filter(
                must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))]
            ),
        ).count or 0)
