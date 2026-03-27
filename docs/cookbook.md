# Cookbook

15+ ready-to-run recipes for common llm-std-lib patterns.

---

## 1. Simple completion

```python
import asyncio
from llm_std_lib import LLMClient

async def main():
    client = LLMClient.from_env()
    r = await client.acomplete(prompt="Summarise the Python GIL in one paragraph.")
    print(r.text)

asyncio.run(main())
```

---

## 2. Sync completion (scripts, notebooks)

`complete()` runs the async call internally — do not call it from inside an
already-running event loop.

```python
from llm_std_lib import LLMClient

client = LLMClient.from_env()
response = client.complete("What is the capital of France?")
print(response.text)   # Paris
```

---

## 3. System prompt

```python
response = await client.acomplete(
    prompt="What time is it?",
    system_prompt="You are a helpful assistant. Always answer in French.",
    model="anthropic/claude-haiku-3",
)
```

---

## 4. Temperature and max tokens

```python
response = await client.acomplete(
    prompt="Write a creative product name for a coffee brand.",
    model="openai/gpt-4o-mini",
    temperature=1.2,
    max_tokens=50,
)
```

---

## 5. Semantic cache with Redis

```python
from llm_std_lib import LLMClient, LLMConfig, SemanticCache
from llm_std_lib.cache.backends.redis import RedisBackend
from llm_std_lib.cache.encoders.openai import OpenAIEncoder

cache = SemanticCache(
    backend=RedisBackend(url="redis://localhost:6379"),
    encoder=OpenAIEncoder(api_key="sk-..."),
    similarity_threshold=0.92,
    ttl=3600,
)
config = LLMConfig.from_env()
client = LLMClient(config.model_copy(update={"cache": cache}))

r1 = await client.acomplete(prompt="What is Python?")
r2 = await client.acomplete(prompt="Tell me about Python programming.")
print(r2.cached)  # True — served from cache
```

---

## 6. Local (offline) cache with sentence-transformers

No API calls for embeddings — runs entirely on-device.

```bash
pip install "llm-std-lib[local-embeddings]"
```

```python
from llm_std_lib import LLMClient, LLMConfig, SemanticCache
from llm_std_lib.cache.backends.memory import MemoryBackend
from llm_std_lib.cache.encoders.local import LocalEncoder

cache = SemanticCache(
    backend=MemoryBackend(),
    encoder=LocalEncoder(model_name="all-MiniLM-L6-v2"),
    similarity_threshold=0.88,
)
client = LLMClient(LLMConfig.from_env().model_copy(update={"cache": cache}))
```

---

## 7. Cache with ChromaDB backend

```bash
pip install "llm-std-lib[chroma]"
```

```python
from llm_std_lib.cache.backends.chroma import ChromaBackend
from llm_std_lib.cache.encoders.openai import OpenAIEncoder

cache = SemanticCache(
    backend=ChromaBackend(host="localhost", port=8000),
    encoder=OpenAIEncoder(api_key="sk-..."),
    similarity_threshold=0.90,
)
```

---

## 8. Cache with AES-256 encryption

```bash
pip install "llm-std-lib[encryption]"
```

```python
import os
from llm_std_lib import LLMConfig, SemanticCache
from llm_std_lib.cache.backends.redis import RedisBackend
from llm_std_lib.cache.encoders.openai import OpenAIEncoder
from llm_std_lib.cache.encryption import CacheEncryption

key_b64 = os.environ.get("CACHE_KEY")
enc = CacheEncryption.from_base64(key_b64) if key_b64 else CacheEncryption.generate()
print("Save this key:", enc.export_key())

cache = SemanticCache(
    backend=RedisBackend(url="redis://localhost:6379"),
    encoder=OpenAIEncoder(api_key="sk-..."),
    encryption=enc,
)
```

---

## 9. Tag-based cache invalidation

```python
from llm_std_lib.cache.backends.base import CacheEntry
import numpy as np

backend = cache.backend
encoder = cache.encoder

vec = await encoder.encode("List EU capitals")
entry = CacheEntry(
    key="eu-capitals",
    prompt="List EU capitals",
    response_text="Paris, Berlin, Rome, ...",
    vector=vec,
    namespace="geo",
    tags={"geo", "capitals"},
)
await backend.store(entry)

# Later — invalidate everything tagged "geo"
removed = await backend.invalidate_by_tag("geo", namespace="geo")
print(f"Removed {removed} entries")
```

---

## 10. Complexity-based routing

```python
from llm_std_lib import LLMClient, LLMConfig, ModelRouter

router = ModelRouter.complexity_based(tiers=[
    {"models": ["openai/gpt-4o-mini"],      "max_complexity": 0.25},
    {"models": ["openai/gpt-4o"],           "max_complexity": 0.65},
    {"models": ["anthropic/claude-opus-3"], "max_complexity": 1.0},
])

# Simple greeting → gpt-4o-mini; PhD thesis analysis → claude-opus-3
client = LLMClient(LLMConfig.from_env(), router=router)
```

---

## 11. Cost-optimised routing

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

## 12. Latency-optimised routing

```python
router = ModelRouter.latency_optimized(
    models=["openai/gpt-4o-mini", "groq/llama-3.1-8b-instant"],
    window_size=50,   # rolling window for p50 latency
)
```

---

## 13. Round-robin routing

```python
router = ModelRouter.round_robin(
    models=["openai/gpt-4o-mini", "anthropic/claude-haiku-3"]
)
client = LLMClient(LLMConfig.from_env(), router=router)
```

---

## 14. Provider fallback chain

```python
from llm_std_lib.resilience.fallback import FallbackChain
from llm_std_lib.resilience.retry import RetryPolicy
from llm_std_lib.providers.openai import OpenAIProvider
from llm_std_lib.providers.anthropic import AnthropicProvider
from llm_std_lib.config import LLMConfig, ProviderConfig

config = LLMConfig.from_env()
chain = FallbackChain(
    providers=[
        OpenAIProvider(config.providers["openai"]),
        AnthropicProvider(config.providers["anthropic"]),
    ],
    retry_policy=RetryPolicy(max_attempts=2, base_delay=0.5),
    failure_threshold_ratio=0.5,
    recovery_timeout=30.0,
)

from llm_std_lib.types import RequestContext
ctx = RequestContext(prompt="Hello", model="gpt-4o-mini", provider="openai")
response = await chain.execute(ctx)
```

---

## 15. Retry with circuit breaker

```python
from llm_std_lib.resilience import ResilienceEngine
from llm_std_lib.resilience.circuit_breaker import CircuitBreaker
from llm_std_lib.resilience.retry import RetryPolicy
from llm_std_lib.resilience.backend import InMemoryBackend

engine = ResilienceEngine(
    breaker=CircuitBreaker(
        backend=InMemoryBackend(),
        key="openai",
        failure_threshold_ratio=0.5,
        recovery_timeout=30.0,
    ),
    retryer=RetryPolicy(max_attempts=3, base_delay=1.0),
)
response = await engine.execute(
    lambda: client.acomplete(prompt="Hello", model="openai/gpt-4o-mini")
)
```

---

## 16. PII redaction middleware

```python
from llm_std_lib import LLMClient, LLMConfig
from llm_std_lib.middleware.builtins.pii import PIIRedactorMiddleware
from llm_std_lib.types import RequestContext

client = LLMClient(LLMConfig.from_env())
pii = PIIRedactorMiddleware()

ctx = RequestContext(
    prompt="My email is alice@example.com, summarise this.",
    model="gpt-4o-mini",
    provider="openai",
)
ctx = await pii.pre_request(ctx)   # email is redacted before dispatch
response_ctx = await client._dispatch(ctx)
```

---

## 17. Cost tracking middleware

```python
from llm_std_lib import LLMClient, LLMConfig
from llm_std_lib.middleware.builtins.cost import CostTrackerMiddleware
from llm_std_lib.types import RequestContext

client = LLMClient(LLMConfig.from_env())
tracker = CostTrackerMiddleware()

for i in range(5):
    ctx = RequestContext(
        prompt=f"Question {i}", model="gpt-4o-mini", provider="openai"
    )
    response_ctx = await client._dispatch(ctx)
    await tracker.post_request(ctx, response_ctx)

print(f"Total cost: ${tracker.total_cost:.6f}")
```

---

## 18. Custom middleware

```python
from llm_std_lib.middleware.base import BaseMiddleware
from llm_std_lib.types import RequestContext, ResponseContext

class AuditMiddleware(BaseMiddleware):
    async def pre_request(self, ctx: RequestContext) -> RequestContext:
        print(f"[AUDIT] prompt={ctx.prompt[:60]!r}")
        return ctx

    async def post_request(
        self, ctx: RequestContext, response: ResponseContext
    ) -> ResponseContext:
        print(f"[AUDIT] tokens={response.total_tokens}, cost=${response.cost_usd:.6f}")
        return response
```

---

## 19. Prometheus metrics

```bash
pip install "llm-std-lib[prometheus]"
```

```python
from llm_std_lib.metrics import MetricsCollector, PrometheusExporter
from llm_std_lib.types import RequestContext

exporter  = PrometheusExporter()
collector = MetricsCollector(on_record=exporter.update)
exporter.start_http_server(port=8000)   # scrape at http://localhost:8000/metrics

ctx = RequestContext(prompt="Hello", model="gpt-4o-mini", provider="openai")
response_ctx = await client._dispatch(ctx)
await collector.post_request(ctx, response_ctx)
```

Import `grafana_dashboard.json` (repo root) into Grafana for a ready-made dashboard.

---

## 20. OpenTelemetry / OTLP

```bash
pip install "llm-std-lib[otlp]"
```

```python
from llm_std_lib.metrics import MetricsCollector, OTLPExporter

exporter  = OTLPExporter(endpoint="http://localhost:4317", service_name="my-app")
collector = MetricsCollector(on_record=exporter.update)

import atexit
atexit.register(exporter.shutdown)
```

---

## 21. Load config from YAML

```yaml
# config.yaml
default_model: openai/gpt-4o-mini
providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    timeout_ms: 15000
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
```

```python
from llm_std_lib import LLMClient, LLMConfig

config = LLMConfig.from_file("config.yaml")
client = LLMClient(config)
```

---

## 22. HashiCorp Vault secrets

```bash
pip install "llm-std-lib[vault]"
```

```python
from llm_std_lib.config import VaultConfig

vault  = VaultConfig(url="https://vault.example.com", secret_path="llm/prod")
config = vault.load()
client = LLMClient(config)
```

---

## 23. Batch processing

```python
import asyncio
from llm_std_lib import LLMClient

client = LLMClient.from_env()
prompts = ["Summarise: " + text for text in documents]

responses = await asyncio.gather(
    *[client.acomplete(prompt=p, model="openai/gpt-4o-mini") for p in prompts]
)

for r in responses:
    print(r.text[:80])
```
