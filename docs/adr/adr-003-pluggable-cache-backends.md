# ADR-003: Pluggable cache backends via abstract base class

**Status:** Accepted
**Date:** 2025-01-20

## Context

Different deployment environments require different vector stores:
- Cloud deployments → Pinecone, Qdrant
- On-premise → ChromaDB, Redis
- Testing → in-memory

Hard-coding any one backend makes the library unsuitable for half the target audience.

## Decision

`BaseCacheBackend` defines an abstract interface (`search`, `store`, `delete`, `clear`,
`invalidate_by_tag`, `size`). Each backend is an optional extra dependency. The library
ships four concrete implementations; users can add their own by subclassing.

## Consequences

- **Positive:** Users choose the backend that fits their stack; no forced dependencies.
- **Positive:** Easy to test with `InMemoryBackend` without standing up infrastructure.
- **Negative:** Adds interface complexity; each backend must maintain parity with the spec.
