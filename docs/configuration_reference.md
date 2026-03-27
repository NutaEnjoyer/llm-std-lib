# Configuration Reference

## LLMConfig

`llm_std_lib.LLMConfig` is the root configuration object. All fields are optional
unless noted.

### Constructor

```python
LLMConfig(
    providers: dict[str, dict | ProviderConfig] = {},
    default_provider: str | None = None,
    default_model: str | None = None,
)
```

### ProviderConfig fields

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `api_key` | `str \| None` | `None` | API key for the provider |
| `base_url` | `str \| None` | provider default | Override the API base URL |
| `timeout_ms` | `int` | `30000` | Request timeout in milliseconds |
| `extra` | `dict` | `{}` | Provider-specific extras (see below) |

### Provider-specific extras

**Azure OpenAI** (`extra` dict):

| Key | Required | Description |
| --- | --- | --- |
| `deployment` | yes | Azure deployment name |
| `api_version` | no | API version (default `2024-02-01`) |

**AWS Bedrock** (`extra` dict):

| Key | Required | Description |
| --- | --- | --- |
| `region` | no | AWS region (default `us-east-1`) |

---

## Loading configuration

### From environment variables

```python
config = LLMConfig.from_env()
```

Reads the following variables:

| Variable | Provider | Field |
| --- | --- | --- |
| `OPENAI_API_KEY` | openai | `api_key` |
| `OPENAI_BASE_URL` | openai | `base_url` |
| `ANTHROPIC_API_KEY` | anthropic | `api_key` |
| `GOOGLE_API_KEY` | google | `api_key` |
| `GROQ_API_KEY` | groq | `api_key` |
| `AZURE_OPENAI_API_KEY` | azure | `api_key` |
| `AZURE_OPENAI_ENDPOINT` | azure | `base_url` |
| `AZURE_OPENAI_DEPLOYMENT` | azure | `extra.deployment` |
| `OLLAMA_BASE_URL` | ollama | `base_url` |
| `LM_STUDIO_BASE_URL` | lm_studio | `base_url` |
| `LM_STUDIO_API_KEY` | lm_studio | `api_key` |
| `AWS_BEDROCK_ENABLED` | bedrock | enables provider |
| `AWS_BEDROCK_REGION` | bedrock | `extra.region` |

### From a YAML file

```yaml
# llm_config.yaml
providers:
  openai:
    api_key: "sk-..."
    timeout_ms: 15000
  anthropic:
    api_key: "sk-ant-..."
```

```python
config = LLMConfig.from_file("llm_config.yaml")
```

Requires `pyyaml`: `pip install pyyaml`.

### From a TOML file

```toml
# llm_config.toml
[providers.openai]
api_key = "sk-..."

[providers.anthropic]
api_key = "sk-ant-..."
```

```python
config = LLMConfig.from_file("llm_config.toml")
```

Uses `tomllib` (built-in on Python 3.11+) or `tomli` on 3.10.

### From HashiCorp Vault

```python
from llm_std_lib.config import VaultConfig

vault = VaultConfig(
    url="https://vault.example.com",
    secret_path="secret/llm/prod",   # KV v2 path
    token="hvs.xxx",                 # or set VAULT_TOKEN env var
    mount_point="secret",
)
config = vault.load()
```

Requires `hvac`: `pip install "llm-std-lib[vault]"`.

---

## SemanticCache

```python
SemanticCache(
    backend: BaseCacheBackend,
    encoder: BaseEncoder,
    threshold: float = 0.90,   # cosine similarity, 0.0–1.0
    ttl: int | None = None,    # seconds; None = no expiry
    namespace: str = "default",
)
```

### Backends

| Extra | Class | Notes |
| --- | --- | --- |
| `redis` | `RedisBackend` | Requires `redis-py` and `numpy` |
| `chroma` | `ChromaBackend` | Ephemeral, persistent, or HTTP client |
| `qdrant` | `QdrantBackend` | In-memory or remote |
| `pinecone` | `PineconeBackend` | Managed cloud |

### Encoders

| Extra | Class | Notes |
| --- | --- | --- |
| *(core)* | `OpenAIEncoder` | `text-embedding-3-small` by default |
| `local-embeddings` | `SentenceTransformerEncoder` | Fully offline |

### Cache encryption

```python
from llm_std_lib.cache.encryption import CacheEncryption

enc = CacheEncryption.generate()     # random AES-256 key
# or
enc = CacheEncryption.from_base64(os.environ["CACHE_KEY"])

cache = SemanticCache(backend=backend, encoder=encoder, encryption=enc)
```

Requires `cryptography`: `pip install "llm-std-lib[encryption]"`.

---

## ModelRouter

```python
# Complexity-based
ModelRouter.complexity_based(tiers=[
    {"models": ["openai/gpt-4o-mini"], "max_complexity": 0.3},
    {"models": ["openai/gpt-4o"],      "max_complexity": 1.0},
])

# Cost-optimised
ModelRouter.cost_optimized(
    models=["openai/gpt-4o-mini", "groq/llama-3.1-70b-versatile"],
    quality_threshold=0.0,
)

# Latency-optimised (tracks rolling p50 latency per model)
ModelRouter.latency_optimized(
    models=["openai/gpt-4o-mini", "groq/llama-3.1-70b-versatile"],
    window_size=100,
)

# Round-robin
ModelRouter.round_robin(models=["openai/gpt-4o-mini", "anthropic/claude-haiku-3"])
```

---

## ResilienceEngine

```python
ResilienceEngine(
    retries: int = 3,
    backoff_base: float = 0.5,    # seconds; doubles each attempt
    backoff_max: float = 60.0,
    timeout_ms: int = 30_000,
    circuit_breaker: CircuitBreaker | None = None,
)
```

```python
CircuitBreaker(
    failure_threshold: int = 5,    # failures before opening
    recovery_timeout: int = 60,    # seconds before half-open
)
```

---

## Metrics exporters

### PrometheusExporter

```python
PrometheusExporter(
    namespace: str = "llm",
    registry: CollectorRegistry | None = None,   # default registry if None
)
exporter.start_http_server(port=8000)
```

### OTLPExporter

```python
OTLPExporter(
    endpoint: str = "http://localhost:4317",
    service_name: str = "llm-std-lib",
    export_interval_ms: int = 30_000,
)
exporter.shutdown()   # flush before process exit
```
