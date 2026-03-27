# Cookbook

15+ ready-to-run recipes for common llm-std-lib patterns.

---

## 1. Simple completion

```python
import asyncio
from llm_std_lib import LLMClient, LLMConfig

async def main():
    client = LLMClient(config=LLMConfig.from_env())
    r = await client.acomplete(prompt="Summarise the Python GIL in one paragraph.")
    print(r.text)

asyncio.run(main())
```

---

## 2. Streaming output

```python
async def stream_response():
    client = LLMClient(config=LLMConfig.from_env())
    async for chunk in client.astream(
        prompt="Write a haiku about async Python.",
        model="openai/gpt-4o-mini",
    ):
        print(chunk, end="", flush=True)
    print()
```

---

## 3. Multi-turn chat

```python
from llm_std_lib.types import RequestContext

history = [
    {"role": "user",      "content": "My name is Alice."},
    {"role": "assistant", "content": "Hi Alice! How can I help?"},
    {"role": "user",      "content": "What is my name?"},
]

ctx = RequestContext(
    prompt=history[-1]["content"],
    messages=history[:-1],   # prior turns
    model="openai/gpt-4o-mini",
)
response = await client.acomplete_ctx(ctx)
print(response.text)  # → Alice
```

---

## 4. System prompt

```python
response = await client.acomplete(
    prompt="What time is it?",
    system_prompt="You are a helpful assistant. Always answer in French.",
    model="anthropic/claude-haiku-3",
)
```

---

## 5. Semantic cache with Redis

```python
from llm_std_lib.cache import SemanticCache
from llm_std_lib.cache.backends.redis import RedisBackend
from llm_std_lib.cache.encoders.openai import OpenAIEncoder

cache = SemanticCache(
    backend=RedisBackend(url="redis://localhost:6379", ttl=3600),
    encoder=OpenAIEncoder(api_key="sk-..."),
    threshold=0.92,
)
client = LLMClient(config=config, cache=cache)

r = await client.acomplete(prompt="What is Python?")
print(r.from_cache)  # False on first call, True on subsequent similar calls
```

---

## 6. Local (offline) cache with ChromaDB

```python
from llm_std_lib.cache.backends.chroma import ChromaBackend
from llm_std_lib.cache.encoders.sentence_transformer import SentenceTransformerEncoder

cache = SemanticCache(
    backend=ChromaBackend(persist_directory="./chroma_data"),
    encoder=SentenceTransformerEncoder(model="all-MiniLM-L6-v2"),
    threshold=0.88,
)
```

---

## 7. Cache with AES-256 encryption

```python
import os
from llm_std_lib.cache.encryption import CacheEncryption

key = os.environ.get("CACHE_KEY")
enc = CacheEncryption.from_base64(key) if key else CacheEncryption.generate()
print("Save this key:", enc.export_key())

cache = SemanticCache(backend=backend, encoder=encoder, encryption=enc)
```

---

## 8. Tag-based cache invalidation

```python
# Store with tags
await cache.set(
    prompt="List EU capitals",
    response="Paris, Berlin, ...",
    tags={"geo", "capitals"},
)

# Later — invalidate everything tagged "geo"
removed = await cache.backend.invalidate_by_tag("geo")
print(f"Removed {removed} entries")
```

---

## 9. Complexity-based routing

```python
from llm_std_lib import ModelRouter

router = ModelRouter.complexity_based(tiers=[
    {"models": ["openai/gpt-4o-mini"],      "max_complexity": 0.25},
    {"models": ["openai/gpt-4o"],           "max_complexity": 0.65},
    {"models": ["anthropic/claude-opus-3"], "max_complexity": 1.0},
])

# Simple greeting → gpt-4o-mini; PhD thesis analysis → claude-opus-3
client = LLMClient(config=config, router=router)
```

---

## 10. Cost-optimised routing

```python
router = ModelRouter.cost_optimized(
    models=[
        "openai/gpt-4o-mini",
        "groq/llama-3.1-70b-versatile",
        "anthropic/claude-haiku-3",
    ]
)
```

---

## 11. Latency-optimised routing

```python
router = ModelRouter.latency_optimized(
    models=["openai/gpt-4o-mini", "groq/llama-3.1-8b-instant"],
    window_size=50,   # rolling window for p50 latency
)
```

---

## 12. Provider fallback chain

```python
from llm_std_lib.resilience.fallback import FallbackChain

chain = FallbackChain(providers=[
    client._build_provider("openai"),
    client._build_provider("anthropic"),
    client._build_provider("groq"),
])

ctx = RequestContext(prompt="Hello", model="gpt-4o-mini")
response = await chain.execute(ctx)
```

---

## 13. Retry with circuit breaker

```python
from llm_std_lib.resilience import ResilienceEngine
from llm_std_lib.resilience.circuit_breaker import CircuitBreaker

engine = ResilienceEngine(
    retries=4,
    backoff_base=1.0,
    timeout_ms=20_000,
    circuit_breaker=CircuitBreaker(failure_threshold=3, recovery_timeout=30),
)
response = await engine.execute(lambda: client.acomplete(prompt="Hello"))
```

---

## 14. PII redaction middleware

```python
from llm_std_lib.middleware.builtins.injection import PromptInjectionGuard

guard = PromptInjectionGuard()  # blocks prompt injection attempts
client = LLMClient(config=config, middleware=[guard])
```

---

## 15. Structured cost tracking

```python
from llm_std_lib.metrics import MetricsCollector

collector = MetricsCollector()
client = LLMClient(config=config, middleware=[collector])

for _ in range(10):
    await client.acomplete(prompt="Hello", model="openai/gpt-4o-mini")

snap = collector.snapshot()
print(f"Total cost: ${snap.cost_usd_total:.4f}")
print(f"Cache hit rate: {snap.cache_hit_rate:.1%}")
print(f"P95 latency: {snap.latency_p95_ms:.0f}ms")
```

---

## 16. Prometheus + Grafana

```bash
pip install "llm-std-lib[prometheus]"
```

```python
from llm_std_lib.metrics import MetricsCollector, PrometheusExporter

exporter  = PrometheusExporter()
collector = MetricsCollector(on_record=exporter.update)
exporter.start_http_server(port=8000)   # → http://localhost:8000/metrics

client = LLMClient(config=config, middleware=[collector])
```

Import `grafana_dashboard.json` (repo root) into Grafana for a ready-made dashboard.

---

## 17. OpenTelemetry / OTLP

```bash
pip install "llm-std-lib[otlp]"
```

```python
from llm_std_lib.metrics import MetricsCollector, OTLPExporter

exporter  = OTLPExporter(endpoint="http://localhost:4317", service_name="my-app")
collector = MetricsCollector(on_record=exporter.update)

client = LLMClient(config=config, middleware=[collector])

import atexit
atexit.register(exporter.shutdown)
```

---

## 18. Load config from YAML with Vault override

```python
from llm_std_lib import LLMConfig
from llm_std_lib.config import VaultConfig

base   = LLMConfig.from_file("llm_config.yaml")
vault  = VaultConfig(url="https://vault.example.com", secret_path="llm/prod")
config = vault.load(base_config=base)   # Vault secrets override base values
```

---

## 19. Batch processing

```python
import asyncio

prompts = ["Summarise: " + text for text in documents]

responses = await asyncio.gather(
    *[client.acomplete(prompt=p, model="openai/gpt-4o-mini") for p in prompts]
)

for r in responses:
    print(r.text[:80])
```

---

## 20. Custom middleware

```python
from llm_std_lib.middleware.base import BaseMiddleware
from llm_std_lib.types import RequestContext, ResponseContext

class AuditMiddleware(BaseMiddleware):
    async def before_request(self, ctx: RequestContext) -> RequestContext:
        print(f"[AUDIT] prompt={ctx.prompt[:60]!r}")
        return ctx

    async def after_response(
        self, ctx: RequestContext, response: ResponseContext
    ) -> ResponseContext:
        print(f"[AUDIT] tokens={response.total_tokens}, cost=${response.cost_usd:.6f}")
        return response

client = LLMClient(config=config, middleware=[AuditMiddleware()])
```
