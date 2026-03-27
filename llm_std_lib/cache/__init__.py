"""
Cache sub-package for llm_std_lib.

Provides semantic caching of LLM responses backed by pluggable vector-store
backends (in-memory, Redis, Chroma, Qdrant, Pinecone) and configurable
text encoders.  Optional AES-256-GCM encryption via :class:`CacheEncryption`.
"""

from .encryption import CacheEncryption

__all__ = ["CacheEncryption"]
