# llm-std-lib

**Python infrastructure library for production-grade LLM API calls.**

[![CI](https://github.com/NutaEnjoyer/llm-std-lib/actions/workflows/ci.yml/badge.svg)](https://github.com/NutaEnjoyer/llm-std-lib/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/NutaEnjoyer/llm-std-lib/branch/main/graph/badge.svg)](https://codecov.io/gh/NutaEnjoyer/llm-std-lib)
[![PyPI](https://img.shields.io/pypi/v/llm-std-lib.svg)](https://pypi.org/project/llm-std-lib/)
[![Python](https://img.shields.io/pypi/pyversions/llm-std-lib.svg)](https://pypi.org/project/llm-std-lib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

One unified async client for **OpenAI, Anthropic, Google, Azure OpenAI, AWS Bedrock, Ollama, Groq, and LM Studio** — with semantic caching, model routing, resilience, and Prometheus/OTLP observability built in.

---

## Features

| Feature | Description |
| --- | --- |
| **Unified client** | `LLMClient.acomplete()` works across 8 providers with one API |
| **Semantic cache** | Avoid re-paying for similar prompts (Redis, ChromaDB, Qdrant, Pinecone) |
| **Model router** | Route by complexity, cost, latency, or round-robin |
| **Resilience** | Retry, circuit breaker, timeout, and provider fallback chains |
| **Middleware** | PII redaction, cost tracking, rate limiting — all pluggable |
| **Observability** | Prometheus and OpenTelemetry (OTLP) exporters, Grafana dashboard included |
| **Flexible config** | YAML/TOML files, environment variables, HashiCorp Vault, AES-256 encryption |

---

## Installation

```bash
# Core only
pip install llm-std-lib

# With common extras
pip install "llm-std-lib[redis,prometheus]"

# Everything
pip install "llm-std-lib[all]"
```

Available extras: `redis`, `chroma`, `qdrant`, `pinecone`, `prometheus`, `otlp`,
`local-embeddings`, `vault`, `encryption`, `all`.

---

## Quick start

```python
import asyncio
from llm_std_lib import LLMClient, LLMConfig

config = LLMConfig(
    default_model="openai/gpt-4o-mini",
    providers={"openai": {"api_key": "sk-..."}},
)
client = LLMClient(config)

async def main():
    response = await client.acomplete(
        prompt="Explain semantic caching in one sentence.",
    )
    print(response.text)
    print(f"Cost: ${response.cost_usd:.6f}  |  Latency: {response.latency_ms:.0f}ms")

asyncio.run(main())
```

Or even shorter — load keys from environment variables:

```bash
export OPENAI_API_KEY=sk-...
```

```python
client = LLMClient.from_env()
response = client.complete("What is the capital of France?")  # sync
print(response.text)  # Paris
```

---

## Semantic cache

```python
from llm_std_lib import LLMClient, LLMConfig, SemanticCache
from llm_std_lib.cache.backends.redis import RedisBackend
from llm_std_lib.cache.encoders.openai import OpenAIEncoder

cache = SemanticCache(
    backend=RedisBackend(url="redis://localhost:6379"),
    encoder=OpenAIEncoder(api_key="sk-..."),
    similarity_threshold=0.92,
)

config = LLMConfig(
    default_model="openai/gpt-4o-mini",
    providers={"openai": {"api_key": "sk-..."}},
    cache=cache,
)
client = LLMClient(config)

r1 = await client.acomplete(prompt="What is the capital of France?")
r2 = await client.acomplete(prompt="Tell me the capital city of France.")
print(r2.cached)   # True — served from cache
```

---

## Model router

```python
from llm_std_lib import LLMClient, LLMConfig, ModelRouter

router = ModelRouter.complexity_based(tiers=[
    {"models": ["openai/gpt-4o-mini"],      "max_complexity": 0.3},
    {"models": ["openai/gpt-4o"],           "max_complexity": 0.7},
    {"models": ["anthropic/claude-opus-3"], "max_complexity": 1.0},
])

client = LLMClient(LLMConfig.from_env(), router=router)

# Model is selected automatically based on prompt complexity
r = await client.acomplete(prompt="Hi")                              # → gpt-4o-mini
r = await client.acomplete(prompt="Prove Fermat's Last Theorem")     # → claude-opus-3
```

---

## Resilience

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
    retryer=RetryPolicy(max_attempts=3, base_delay=0.5),
)

response = await engine.execute(
    lambda: client.acomplete(prompt="Hello")
)
```

---

## Observability

```python
from llm_std_lib.metrics import MetricsCollector, PrometheusExporter
from llm_std_lib.types import RequestContext, ResponseContext

exporter  = PrometheusExporter()
collector = MetricsCollector(on_record=exporter.update)
exporter.start_http_server(port=8000)   # scrape at http://localhost:8000/metrics

# Attach manually after each response
ctx = RequestContext(prompt="Hello", model="gpt-4o-mini", provider="openai")
response_ctx = await client._dispatch(ctx)
await collector.post_request(ctx, response_ctx)
```

Import `grafana_dashboard.json` (included in the repo) for a ready-made Grafana dashboard.

---

## Supported providers

| Provider | Model prefix | Auth env var |
| --- | --- | --- |
| OpenAI | `openai/` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/` | `ANTHROPIC_API_KEY` |
| Google Gemini | `google/` | `GOOGLE_API_KEY` |
| Azure OpenAI | `azure/` | `AZURE_OPENAI_API_KEY` |
| AWS Bedrock | `bedrock/` | AWS credential chain |
| Ollama (local) | `ollama/` | — |
| Groq | `groq/` | `GROQ_API_KEY` |
| LM Studio (local) | `lm_studio/` | optional |

---

## Configuration

### From environment variables

```python
from llm_std_lib import LLMClient

client = LLMClient.from_env()
```

### From a YAML file

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
from llm_std_lib import LLMConfig

config = LLMConfig.from_file("config.yaml")
```

### From HashiCorp Vault

```python
from llm_std_lib.config import VaultConfig

vault  = VaultConfig(url="https://vault.example.com", secret_path="llm/prod")
config = vault.load()
```

---

## Documentation

- [Getting Started Guide](docs/getting_started.md)
- [Configuration Reference](docs/configuration_reference.md)
- [Cookbook](docs/cookbook.md) — 20 recipes
- [Migration Guide](docs/migration_guide.md) — from LiteLLM, LangChain, raw API calls
- [Contributing](CONTRIBUTING.md)

---

## License

MIT
