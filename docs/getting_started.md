# Getting Started

This guide takes you from zero to a working LLM call in under 10 minutes.

## 1. Install

```bash
pip install llm-std-lib
```

## 2. Set your API key

```bash
export OPENAI_API_KEY=sk-...
```

## 3. Make your first call

```python
import asyncio
from llm_std_lib import LLMClient, LLMConfig

async def main():
    config = LLMConfig.from_env()
    client = LLMClient(config=config)

    response = await client.acomplete(
        prompt="What is the capital of France?",
        model="openai/gpt-4o-mini",
    )
    print(response.text)
    # → Paris

asyncio.run(main())
```

`response` is a `ResponseContext` with:

- `response.text` — the model's reply
- `response.cost_usd` — estimated cost
- `response.latency_ms` — round-trip time
- `response.prompt_tokens` / `response.completion_tokens`
- `response.from_cache` — `True` if served from semantic cache

## 4. Add a semantic cache

Re-using answers for similar prompts cuts costs dramatically.

```bash
pip install "llm-std-lib[redis]"
docker run -d -p 6379:6379 redis:7-alpine
```

```python
from llm_std_lib.cache import SemanticCache
from llm_std_lib.cache.backends.redis import RedisBackend
from llm_std_lib.cache.encoders.openai import OpenAIEncoder

cache = SemanticCache(
    backend=RedisBackend(url="redis://localhost:6379"),
    encoder=OpenAIEncoder(api_key="sk-..."),
    threshold=0.92,   # cosine similarity threshold
)

client = LLMClient(config=config, cache=cache)
```

## 5. Route across models

Send cheap requests to a small model and complex ones to a large model automatically:

```python
from llm_std_lib import ModelRouter

router = ModelRouter.complexity_based(tiers=[
    {"models": ["openai/gpt-4o-mini"], "max_complexity": 0.4},
    {"models": ["openai/gpt-4o"],      "max_complexity": 1.0},
])

client = LLMClient(config=config, router=router)
```

## 6. Add resilience

```python
from llm_std_lib.resilience import ResilienceEngine

engine = ResilienceEngine(retries=3, backoff_base=0.5, timeout_ms=15_000)

response = await engine.execute(
    lambda: client.acomplete(prompt="Hello")
)
```

## 7. Export metrics

```bash
pip install "llm-std-lib[prometheus]"
```

```python
from llm_std_lib.metrics import MetricsCollector, PrometheusExporter

exporter  = PrometheusExporter()
collector = MetricsCollector(on_record=exporter.update)
exporter.start_http_server(port=8000)

client = LLMClient(config=config, middleware=[collector])
```

Prometheus will scrape `http://localhost:8000/metrics`.

## Next steps

- [Configuration Reference](configuration_reference.md) — all options explained
- [Cookbook](cookbook.md) — 15+ real-world recipes
- [Migration Guide](migration_guide.md) — coming from LiteLLM or LangChain
