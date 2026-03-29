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
from llm_std_lib import LLMClient

async def main():
    client = LLMClient.from_env()

    response = await client.acomplete(
        prompt="What is the capital of France?",
        model="openai/gpt-4o-mini",
    )
    print(response.text)
    # → Paris

asyncio.run(main())
```

Or synchronously:

```python
client = LLMClient.from_env()
response = client.complete("What is the capital of France?")
print(response.text)
```

`response` is an `LLMResponse` with:

| Field | Type | Description |
| --- | --- | --- |
| `response.text` | `str` | The model's reply |
| `response.cost_usd` | `float` | Estimated cost in USD |
| `response.latency_ms` | `float` | Round-trip time in ms |
| `response.prompt_tokens` | `int` | Input token count |
| `response.completion_tokens` | `int` | Output token count |
| `response.total_tokens` | `int` | Total tokens used |
| `response.cached` | `bool` | `True` if served from semantic cache |
| `response.model` | `str` | Model that served the request |
| `response.provider` | `str` | Provider that served the request |

## 4. Add a semantic cache

Re-using answers for similar prompts cuts costs dramatically.

```bash
pip install "llm-std-lib[redis]"
docker run -d -p 6379:6379 redis:7-alpine
```

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
config_with_cache = config.model_copy(update={"cache": cache})
client = LLMClient(config_with_cache)

r1 = await client.acomplete(prompt="What is the capital of France?")
r2 = await client.acomplete(prompt="Tell me the capital city of France.")
print(r2.cached)  # True — served from cache
```

## 5. Route across models

Send cheap requests to a small model and complex ones to a large model automatically:

```python
from llm_std_lib import LLMClient, LLMConfig, ModelRouter

router = ModelRouter.complexity_based(tiers=[
    {"models": ["openai/gpt-4o-mini"], "max_complexity": 0.4},
    {"models": ["openai/gpt-4o"],      "max_complexity": 1.0},
])

client = LLMClient(LLMConfig.from_env(), router=router)
```

## 6. Add resilience

```python
from llm_std_lib import LLMClient, LLMConfig
from llm_std_lib.resilience import ResilienceEngine
from llm_std_lib.resilience.circuit_breaker import CircuitBreaker
from llm_std_lib.resilience.retry import RetryPolicy
from llm_std_lib.resilience.backend import InMemoryBackend

client = LLMClient(LLMConfig.from_env())
engine = ResilienceEngine(
    breaker=CircuitBreaker(
        backend=InMemoryBackend(),
        key="openai",
        failure_threshold_ratio=0.5,
        recovery_timeout=30.0,
    ),
    retryer=RetryPolicy(max_attempts=3, base_delay=0.5),
)

response = await engine.execute(
    lambda: client.acomplete(prompt="Hello", model="openai/gpt-4o-mini")
)
```

## 7. Export metrics

```bash
pip install "llm-std-lib[prometheus]"
```

```python
from llm_std_lib.metrics import MetricsCollector, PrometheusExporter
from llm_std_lib.types import RequestContext

exporter  = PrometheusExporter()
collector = MetricsCollector(on_record=exporter.update)
exporter.start_http_server(port=8000)

# After each request, record metrics manually:
ctx = RequestContext(prompt="Hello", model="gpt-4o-mini", provider="openai")
response_ctx = await client._dispatch(ctx)
await collector.post_request(ctx, response_ctx)
```

Prometheus will scrape `http://localhost:8000/metrics`.

## 8. PII redaction

Встроенный regex-движок закрывает email, телефон, карты, SSN — без доп. зависимостей:

```python
from llm_std_lib.middleware.builtins.pii import PIIRedactorMiddleware
from llm_std_lib.types import RequestContext

pii = PIIRedactorMiddleware()
ctx = RequestContext(prompt="My email is alice@example.com", model="gpt-4o-mini", provider="openai")
ctx = await pii.pre_request(ctx)
# → "My email is [EMAIL]"
```

Для имён и адресов используй Presidio (NER):

```bash
pip install "llm-std-lib[presidio]"

# Скачай spaCy модели для нужных языков (один раз):
python -m spacy download en_core_web_lg   # английский (~750MB)
python -m spacy download ru_core_news_sm  # русский (~15MB)
```

```python
from llm_std_lib.middleware.builtins.presidio_engine import PresidioPIIEngine

engine = PresidioPIIEngine(languages=["ru", "en"])
pii = PIIRedactorMiddleware(engine=engine, language="ru")
# "Меня зовут Иван Петров" → "Меня зовут <PERSON>"
```

## Next steps

- [Configuration Reference](configuration_reference.md) — all options explained
- [Cookbook](cookbook.md) — 15+ real-world recipes
- [Migration Guide](migration_guide.md) — coming from LiteLLM or LangChain
