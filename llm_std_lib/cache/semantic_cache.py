"""
Semantic cache for LLM responses.

Encodes incoming prompts into dense vectors, queries the configured backend
for semantically similar past queries, and returns cached responses when
cosine similarity exceeds the configured threshold.

Usage::

    from llm_std_lib import SemanticCache
    from llm_std_lib.cache.backends.memory import MemoryBackend
    from llm_std_lib.cache.encoders.openai import OpenAIEncoder

    encoder = OpenAIEncoder(api_key="sk-...")
    cache = SemanticCache(
        backend=MemoryBackend(),
        encoder=encoder,
        similarity_threshold=0.92,
        ttl=3600,
        namespace="prod",
    )

    # Integrate with LLMClient via config:
    config = LLMConfig(cache=cache, ...)
"""

from __future__ import annotations

import time
import uuid

from llm_std_lib._logging import get_logger
from llm_std_lib.cache.backends.base import BaseCacheBackend, CacheEntry
from llm_std_lib.cache.encoders.base import BaseEncoder
from llm_std_lib.exceptions import LLMCacheError
from llm_std_lib.types import RequestContext, ResponseContext

_log = get_logger(__name__)

_DEFAULT_THRESHOLD = 0.92
_DEFAULT_TTL: int | None = None  # No expiry by default
_DEFAULT_NAMESPACE = "default"
_DEFAULT_MAX_ENTRIES = 0  # Unlimited


class SemanticCache:
    """Vector-similarity-based cache for LLM responses.

    Encodes incoming prompts, searches the backend for near-duplicate entries,
    and stores new responses for future reuse.

    Args:
        backend: Storage backend (MemoryBackend, RedisBackend, etc.).
        encoder: Embedding model used to vectorise prompts.
        similarity_threshold: Minimum cosine similarity to count as a cache
            hit (default: ``0.92``).
        ttl: Time-to-live in seconds for cached entries. ``None`` = no expiry.
        namespace: Logical partition for cache isolation (e.g. per-project).
        max_entries: Maximum entries in the backend (``0`` = unlimited).
            Forwarded to the backend if it supports it.

    Example::

        cache = SemanticCache(
            backend=MemoryBackend(),
            encoder=OpenAIEncoder(api_key="sk-..."),
            similarity_threshold=0.92,
            ttl=3600,
            namespace="my-app",
        )
    """

    def __init__(
        self,
        backend: BaseCacheBackend,
        encoder: BaseEncoder,
        similarity_threshold: float = _DEFAULT_THRESHOLD,
        ttl: int | None = _DEFAULT_TTL,
        namespace: str = _DEFAULT_NAMESPACE,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise LLMCacheError(
                f"similarity_threshold must be in [0.0, 1.0], got {similarity_threshold}."
            )
        self._backend = backend
        self._encoder = encoder
        self._threshold = similarity_threshold
        self._ttl = ttl
        self._namespace = namespace
        self._max_entries = max_entries

        # Stats counters (best-effort, not thread-safe for reads, but
        # increments are atomic in CPython)
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Core cache operations
    # ------------------------------------------------------------------

    async def lookup(self, ctx: RequestContext) -> ResponseContext | None:
        """Search the cache for a semantically equivalent past request.

        Args:
            ctx: The current request context (uses ``ctx.prompt``).

        Returns:
            A :class:`~llm_std_lib.types.ResponseContext` with
            ``cached=True`` if a hit is found, or ``None`` on cache miss.

        Raises:
            LLMCacheError: If the backend or encoder raises an unexpected error.
        """
        try:
            vector = await self._encoder.encode(ctx.prompt)
            hits = await self._backend.search(
                vector=vector,
                threshold=self._threshold,
                namespace=self._namespace,
                top_k=1,
            )
        except LLMCacheError:
            raise
        except Exception as exc:
            raise LLMCacheError(f"Cache lookup failed: {exc}") from exc

        if not hits:
            self._misses += 1
            _log.debug("cache miss", namespace=self._namespace, request_id=ctx.request_id)
            return None

        entry = hits[0]
        self._hits += 1
        _log.debug(
            "cache hit",
            namespace=self._namespace,
            request_id=ctx.request_id,
            entry_key=entry.key,
        )
        return ResponseContext(
            request_id=ctx.request_id,
            text=entry.response_text,
            model=entry.metadata.get("model", ""),
            provider=entry.metadata.get("provider", ""),
            prompt_tokens=entry.metadata.get("prompt_tokens", 0),
            completion_tokens=entry.metadata.get("completion_tokens", 0),
            total_tokens=entry.metadata.get("total_tokens", 0),
            cost_usd=entry.metadata.get("cost_usd", 0.0),
            latency_ms=0.0,
            cached=True,
            metadata=entry.metadata,
        )

    async def store(self, ctx: RequestContext, response: ResponseContext) -> None:
        """Persist a new request+response pair in the cache.

        Args:
            ctx: The original request context.
            response: The provider's response to cache.

        Raises:
            LLMCacheError: If the backend write fails.
        """
        try:
            vector = await self._encoder.encode(ctx.prompt)
            expires_at = (time.time() + self._ttl) if self._ttl is not None else None
            entry = CacheEntry(
                key=str(uuid.uuid4()),
                prompt=ctx.prompt,
                response_text=response.text,
                vector=vector,
                metadata={
                    "model": response.model,
                    "provider": response.provider,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": response.total_tokens,
                    "cost_usd": response.cost_usd,
                    **ctx.tags,
                },
                namespace=self._namespace,
                expires_at=expires_at,
                tags=set(ctx.tags.values()),
            )
            await self._backend.store(entry)
        except LLMCacheError:
            raise
        except Exception as exc:
            raise LLMCacheError(f"Cache store failed: {exc}") from exc

        _log.debug(
            "cache stored",
            namespace=self._namespace,
            request_id=ctx.request_id,
            entry_key=entry.key,
        )

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    async def clear(self) -> int:
        """Remove all entries in this cache's namespace.

        Returns:
            Number of entries removed.
        """
        count = await self._backend.clear(namespace=self._namespace)
        _log.info("cache cleared", namespace=self._namespace, removed=count)
        return count

    async def invalidate_by_tag(self, tag: str) -> int:
        """Remove all entries carrying *tag* in this cache's namespace.

        Args:
            tag: Tag string to match.

        Returns:
            Number of entries removed.
        """
        count = await self._backend.invalidate_by_tag(tag, namespace=self._namespace)
        _log.info("cache invalidated by tag", tag=tag, namespace=self._namespace, removed=count)
        return count

    # ------------------------------------------------------------------
    # Stats and introspection
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        """Cache hit rate since instantiation (0.0–1.0).

        Returns ``0.0`` if no requests have been made yet.
        """
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def hits(self) -> int:
        """Total cache hits since instantiation."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total cache misses since instantiation."""
        return self._misses

    async def size(self) -> int:
        """Return the number of entries in this cache's namespace."""
        return await self._backend.size(namespace=self._namespace)

    # ------------------------------------------------------------------
    # Configuration properties
    # ------------------------------------------------------------------

    @property
    def namespace(self) -> str:
        """The namespace this cache operates in."""
        return self._namespace

    @property
    def similarity_threshold(self) -> float:
        """Minimum cosine similarity required for a cache hit."""
        return self._threshold

    @property
    def ttl(self) -> int | None:
        """TTL in seconds for new entries, or ``None`` for no expiry."""
        return self._ttl

    @property
    def backend(self) -> BaseCacheBackend:
        """The underlying storage backend."""
        return self._backend

    @property
    def encoder(self) -> BaseEncoder:
        """The embedding encoder used to vectorise prompts."""
        return self._encoder
