# ADR-001: Async-first public API

**Status:** Accepted
**Date:** 2025-01-15

## Context

LLM API calls are I/O-bound and typically take 500ms–30s. Most production Python services
(FastAPI, Starlette, async workers) are built on asyncio. A synchronous-first library forces
users to run blocking calls in thread pools, adding boilerplate and overhead.

## Decision

All primary methods on `LLMClient`, `SemanticCache`, and cache backends are `async`. A thin
`complete()` / `search()` synchronous wrapper is provided only where explicitly needed by
calling `asyncio.run()` internally.

## Consequences

- **Positive:** No thread-pool overhead; natural integration with FastAPI and async frameworks.
- **Positive:** `asyncio.gather()` makes parallel fan-out trivially efficient.
- **Negative:** Users running in synchronous scripts must call `asyncio.run()`.
- **Negative:** Testing requires `pytest-asyncio`.
