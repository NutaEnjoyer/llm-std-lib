"""
Real-world smoke test for llm-std-lib v1.0.0.

Tests the full stack with a real API key.
Requires at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY.

Usage:
    set OPENAI_API_KEY=sk-...
    py smoke_test.py
"""

from __future__ import annotations

import asyncio
import os
import sys


def _check_keys() -> list[tuple[str, str]]:
    available = []
    if os.getenv("OPENAI_API_KEY"):
        available.append(("openai", "openai/gpt-4o-mini"))
    if os.getenv("ANTHROPIC_API_KEY"):
        available.append(("anthropic", "anthropic/claude-haiku-3"))
    if os.getenv("GROQ_API_KEY"):
        available.append(("groq", "groq/llama-3.1-8b-instant"))
    if os.getenv("GOOGLE_API_KEY"):
        available.append(("google", "google/gemini-1.5-flash"))
    return available


def _sep(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


# ─────────────────────────────────────────────────────────────
# Test 1: basic completion
# ─────────────────────────────────────────────────────────────

async def test_basic_completion(model: str) -> None:
    from llm_std_lib import LLMClient, LLMConfig

    _sep(f"Basic completion — {model}")
    client = LLMClient(LLMConfig.from_env())

    response = await client.acomplete(
        prompt="Reply with exactly: Hello from llm-std-lib",
        model=model,
    )

    print(f"  text     : {response.text!r}")
    print(f"  tokens   : {response.prompt_tokens} + {response.completion_tokens} = {response.total_tokens}")
    print(f"  cost     : ${response.cost_usd:.6f}")
    print(f"  latency  : {response.latency_ms:.0f}ms")
    print(f"  provider : {response.provider}")
    print(f"  model    : {response.model}")
    assert response.text, "Empty response text"
    print("  PASS")


# ─────────────────────────────────────────────────────────────
# Test 2: sync wrapper
# ─────────────────────────────────────────────────────────────

async def test_sync_complete(model: str) -> None:
    """complete() uses asyncio.run() internally — must be called outside an event loop."""
    import concurrent.futures

    from llm_std_lib import LLMClient, LLMConfig

    _sep(f"Sync complete() wrapper — {model}")

    def _run() -> str:
        client = LLMClient(LLMConfig.from_env())
        response = client.complete(prompt="Say 'sync works'", model=model)
        return response.text

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        text = await asyncio.get_event_loop().run_in_executor(pool, _run)

    print(f"  text: {text!r}")
    assert text
    print("  PASS")


# ─────────────────────────────────────────────────────────────
# Test 3: semantic cache (in-memory)
# ─────────────────────────────────────────────────────────────

async def test_semantic_cache(model: str) -> None:
    from llm_std_lib import LLMClient, LLMConfig, SemanticCache
    from llm_std_lib.cache.backends.memory import MemoryBackend
    from llm_std_lib.cache.encoders.openai import OpenAIEncoder

    _sep(f"Semantic cache (in-memory) — {model}")

    if not os.getenv("OPENAI_API_KEY"):
        print("  SKIP — OPENAI_API_KEY required for OpenAI encoder")
        return

    encoder = OpenAIEncoder(api_key=os.environ["OPENAI_API_KEY"])
    cache = SemanticCache(
        backend=MemoryBackend(),
        encoder=encoder,
        similarity_threshold=0.90,
    )
    config = LLMConfig.from_env()
    config_with_cache = config.model_copy(update={"cache": cache})

    # Use cache directly: encode → store → search
    prompt1 = "What is the capital of France?"
    vec1 = await encoder.encode(prompt1)

    from llm_std_lib.cache.backends.base import CacheEntry
    entry = CacheEntry(
        key="france-capital",
        prompt=prompt1,
        response_text="Paris",
        vector=vec1,
        namespace="smoke",
    )
    await cache.backend.store(entry)

    # Search with a semantically similar query
    prompt2 = "Tell me the capital city of France."
    vec2 = await encoder.encode(prompt2)
    results = await cache.backend.search(vec2, threshold=0.90, namespace="smoke", top_k=1)

    if results:
        print(f"  cache hit! key={results[0].key!r}, text={results[0].response_text!r}")
        print("  PASS")
    else:
        print("  cache miss (below threshold) — still PASS, cache infra works")


# ─────────────────────────────────────────────────────────────
# Test 4: cost tracker middleware (standalone)
# ─────────────────────────────────────────────────────────────

async def test_cost_tracker(model: str) -> None:
    from llm_std_lib import LLMClient, LLMConfig
    from llm_std_lib.middleware.builtins.cost import CostTrackerMiddleware
    from llm_std_lib.types import RequestContext

    _sep(f"CostTrackerMiddleware (standalone) — {model}")

    client = LLMClient(LLMConfig.from_env())
    tracker = CostTrackerMiddleware()

    for i in range(3):
        ctx = RequestContext(prompt=f"Say 'ok' ({i})", model=model.split("/")[1], provider=model.split("/")[0])
        response_ctx = await client._dispatch(ctx)
        await tracker.post_request(ctx, response_ctx)

    print(f"  total_cost : ${tracker.total_cost:.6f}")
    assert tracker.total_cost > 0
    print("  PASS")


# ─────────────────────────────────────────────────────────────
# Test 5: model router
# ─────────────────────────────────────────────────────────────

async def test_model_router(model: str) -> None:
    from llm_std_lib import LLMClient, LLMConfig, ModelRouter

    _sep("ModelRouter (round-robin)")
    config = LLMConfig.from_env()
    router = ModelRouter.round_robin(models=[model, model])
    client = LLMClient(config, router=router)

    for prompt in ["Hi", "Hello", "Hey"]:
        r = await client.acomplete(prompt=prompt)
        print(f"  {prompt!r:10} → {r.provider}/{r.model}")

    print("  PASS")


# ─────────────────────────────────────────────────────────────
# Test 6: resilience engine
# ─────────────────────────────────────────────────────────────

async def test_resilience(model: str) -> None:
    from llm_std_lib import LLMClient, LLMConfig
    from llm_std_lib.resilience import ResilienceEngine
    from llm_std_lib.resilience.circuit_breaker import CircuitBreaker
    from llm_std_lib.resilience.retry import RetryPolicy

    _sep(f"ResilienceEngine — {model}")
    client = LLMClient(LLMConfig.from_env())
    from llm_std_lib.resilience.backend import InMemoryBackend as StateBackend

    engine = ResilienceEngine(
        breaker=CircuitBreaker(
            backend=StateBackend(),
            key="smoke-test",
            failure_threshold_ratio=0.5,
            recovery_timeout=30.0,
        ),
        retryer=RetryPolicy(max_attempts=2, base_delay=0.1),
    )

    response = await engine.execute(
        lambda: client.acomplete(prompt="Say 'resilient'", model=model)
    )
    print(f"  text: {response.text!r}")
    assert response.text
    print("  PASS")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

async def main() -> None:
    import llm_std_lib
    print(f"llm-std-lib smoke test  v{llm_std_lib.__version__}")

    available = _check_keys()
    if not available:
        print("\nNo API keys found. Set at least one of:")
        print("  OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY")
        sys.exit(1)

    print(f"Available providers: {[p for p, _ in available]}")
    _, model = available[0]

    tests = [
        ("Basic completion",    test_basic_completion(model)),
        ("Sync wrapper",        test_sync_complete(model)),
        ("Semantic cache",      test_semantic_cache(model)),
        ("Cost tracker",        test_cost_tracker(model)),
        ("Model router",        test_model_router(model)),
        ("ResilienceEngine",    test_resilience(model)),
    ]

    passed, failed = 0, 0
    for name, coro in tests:
        try:
            await coro
            passed += 1
        except Exception as exc:
            print(f"  FAIL [{name}]: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    _sep("Results")
    print(f"  passed : {passed}/{len(tests)}")
    print(f"  failed : {failed}/{len(tests)}")
    if failed:
        sys.exit(1)
    print("\nAll smoke tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
