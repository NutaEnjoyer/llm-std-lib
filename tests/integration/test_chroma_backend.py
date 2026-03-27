"""
Integration tests for the ChromaDB cache backend.

Requires a running ChromaDB instance. Start with:
    docker compose up -d chromadb

Environment variables:
    CHROMA_HOST  — ChromaDB host (default: localhost)
    CHROMA_PORT  — ChromaDB port (default: 8000)
"""

from __future__ import annotations

import os

import numpy as np
import pytest

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))


def _skip_if_no_chroma() -> None:
    try:
        import chromadb as _chromadb  # noqa: F401
    except ImportError:
        pytest.skip("chromadb package not installed")


@pytest.fixture
async def backend():
    _skip_if_no_chroma()
    from llm_std_lib.cache.backends.chroma import ChromaBackend

    b = ChromaBackend(host=CHROMA_HOST, port=CHROMA_PORT)
    yield b
    await b.clear()


@pytest.mark.asyncio
async def test_store_and_search(backend):
    from llm_std_lib.cache.backends.base import CacheEntry

    vec = np.random.rand(128).astype(np.float32)
    vec /= np.linalg.norm(vec)

    entry = CacheEntry(
        key="chroma-integ-1",
        prompt="chroma integration",
        response_text="chroma response",
        vector=vec,
        namespace="integ_test",
    )
    await backend.store(entry)

    results = await backend.search(vec, threshold=0.9, namespace="integ_test")
    assert len(results) >= 1
    assert any(r.key == "chroma-integ-1" for r in results)


@pytest.mark.asyncio
async def test_delete(backend):
    from llm_std_lib.cache.backends.base import CacheEntry

    vec = np.ones(128, dtype=np.float32)
    vec /= np.linalg.norm(vec)

    entry = CacheEntry(
        key="chroma-del",
        prompt="delete",
        response_text="bye",
        vector=vec,
        namespace="integ_test",
    )
    await backend.store(entry)
    deleted = await backend.delete("chroma-del", namespace="integ_test")
    assert deleted is True


@pytest.mark.asyncio
async def test_size(backend):
    from llm_std_lib.cache.backends.base import CacheEntry

    for i in range(2):
        vec = np.random.rand(128).astype(np.float32)
        vec /= np.linalg.norm(vec)
        entry = CacheEntry(
            key=f"chroma-size-{i}",
            prompt=f"p{i}",
            response_text=f"r{i}",
            vector=vec,
            namespace="integ_test",
        )
        await backend.store(entry)

    size = await backend.size(namespace="integ_test")
    assert size >= 2


@pytest.mark.asyncio
async def test_invalidate_by_tag(backend):
    from llm_std_lib.cache.backends.base import CacheEntry

    vec = np.random.rand(128).astype(np.float32)
    vec /= np.linalg.norm(vec)

    entry = CacheEntry(
        key="chroma-tag",
        prompt="tagged",
        response_text="tagged",
        vector=vec,
        namespace="integ_test",
        tags={"chroma-integ-tag"},
    )
    await backend.store(entry)

    removed = await backend.invalidate_by_tag("chroma-integ-tag", namespace="integ_test")
    assert removed >= 1
