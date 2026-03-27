"""
ChromaDB cache backend.

Uses ChromaDB as the vector store for the semantic cache. Recommended for
offline / on-premise deployments where a managed cloud service is undesirable.

Requires the ``chroma`` optional dependency::

    pip install llm-std-lib[chroma]
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from llm_std_lib.cache.backends.base import BaseCacheBackend, CacheEntry
from llm_std_lib.exceptions import LLMCacheError


class ChromaBackend(BaseCacheBackend):
    """ChromaDB-backed semantic cache.

    Each namespace maps to a separate Chroma collection. Similarity search
    is delegated to Chroma's built-in HNSW index.

    Args:
        host: Chroma server host (default: ``localhost``). Pass ``None`` to
            use an in-process ephemeral client (useful for testing).
        port: Chroma server port (default: ``8000``).
        persist_directory: Local path for a persistent in-process client.
            Ignored when *host* is set.

    Raises:
        LLMCacheError: If ``chromadb`` is not installed.

    Example::

        backend = ChromaBackend(host="localhost", port=8000)
        cache = SemanticCache(backend=backend, encoder=encoder)
    """

    def __init__(
        self,
        host: str | None = "localhost",
        port: int = 8000,
        persist_directory: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._persist_dir = persist_directory
        self._client: Any = None
        self._collections: dict[str, Any] = {}

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise LLMCacheError(
                    "chromadb is not installed. Run: pip install llm-std-lib[chroma]"
                ) from exc
            if self._host is not None:
                self._client = chromadb.HttpClient(host=self._host, port=self._port)
            elif self._persist_dir:
                self._client = chromadb.PersistentClient(path=self._persist_dir)
            else:
                self._client = chromadb.EphemeralClient()
        return self._client

    def _get_collection(self, namespace: str) -> Any:
        if namespace not in self._collections:
            client = self._get_client()
            self._collections[namespace] = client.get_or_create_collection(
                name=f"llm_std_lib_{namespace}",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[namespace]

    async def search(
        self,
        vector: np.ndarray,
        threshold: float,
        namespace: str = "default",
        top_k: int = 1,
    ) -> list[CacheEntry]:
        """Query Chroma's HNSW index for the nearest neighbours."""
        collection = self._get_collection(namespace)
        now = time.time()

        results = collection.query(
            query_embeddings=[vector.tolist()],
            n_results=min(top_k * 2, max(1, collection.count())),
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        entries: list[CacheEntry] = []
        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        embeddings = results.get("embeddings", [[]])[0]

        for i, doc_id in enumerate(ids):
            # Chroma cosine distance = 1 - cosine_similarity
            similarity = 1.0 - distances[i]
            if similarity < threshold:
                continue
            meta = metadatas[i] or {}
            expires_at = meta.get("expires_at")
            if expires_at and float(expires_at) <= now:
                continue
            tags = set(json.loads(meta.get("tags", "[]")))
            emb = np.array(embeddings[i], dtype=np.float32) if embeddings[i] is not None else vector
            doc = documents[i] if i < len(documents) else None
            entry = CacheEntry(
                key=doc_id,
                prompt=meta.get("prompt", ""),
                response_text=doc or "",
                vector=emb,
                metadata=json.loads(meta.get("extra_metadata", "{}")),
                namespace=namespace,
                expires_at=float(expires_at) if expires_at else None,
                tags=tags,
            )
            entries.append(entry)
            if len(entries) >= top_k:
                break

        return entries

    async def store(self, entry: CacheEntry) -> None:
        """Upsert *entry* into the Chroma collection."""
        collection = self._get_collection(entry.namespace)
        collection.upsert(
            ids=[entry.key],
            embeddings=[entry.vector.tolist()],
            documents=[entry.response_text],
            metadatas=[{
                "prompt": entry.prompt,
                "expires_at": str(entry.expires_at or ""),
                "tags": json.dumps(list(entry.tags)),
                "extra_metadata": json.dumps(entry.metadata),
            }],
        )

    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete a single entry by its ID."""
        collection = self._get_collection(namespace)
        try:
            collection.delete(ids=[key])
            return True
        except Exception:
            return False

    async def clear(self, namespace: str | None = None) -> int:
        """Delete all entries, optionally filtered by namespace."""
        client = self._get_client()
        removed = 0
        collections_to_clear = (
            [namespace] if namespace else list(self._collections.keys())
        )
        for ns in collections_to_clear:
            col = self._get_collection(ns)
            count = col.count()
            client.delete_collection(f"llm_std_lib_{ns}")
            self._collections.pop(ns, None)
            removed += count
        return removed

    async def invalidate_by_tag(self, tag: str, namespace: str | None = None) -> int:
        """Delete all entries that carry *tag*."""
        namespaces = [namespace] if namespace else list(self._collections.keys())
        removed = 0
        for ns in namespaces:
            col = self._get_collection(ns)
            results = col.get(include=["metadatas"])
            ids_to_delete = [
                doc_id for doc_id, meta in zip(results["ids"], results["metadatas"])
                if tag in json.loads((meta or {}).get("tags", "[]"))
            ]
            if ids_to_delete:
                col.delete(ids=ids_to_delete)
                removed += len(ids_to_delete)
        return removed

    async def size(self, namespace: str | None = None) -> int:
        """Return the total number of entries."""
        namespaces = [namespace] if namespace else list(self._collections.keys())
        return sum(self._get_collection(ns).count() for ns in namespaces)
