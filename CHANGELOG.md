# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-03-27

### Added

#### Core client

- `LLMClient` — unified async entry point for all provider interactions.
- `LLMConfig` — Pydantic-based configuration with `from_env()`, `from_file()` (YAML/TOML).
- `VaultConfig` — HashiCorp Vault KV v2 secrets integration.
- `RequestContext`, `ResponseContext` — normalised type layer.
- Exception hierarchy: `LLMError`, `LLMProviderError`, `LLMRateLimitError`,
  `LLMTimeoutError`, `LLMAllFallbacksFailedError`, `LLMCircuitOpenError`,
  `LLMConfigError`, `LLMCacheError`, `LLMMiddlewareError`, `LLMValidationError`.

#### Provider adapters (8 providers via httpx — no vendor SDKs required)

- `OpenAIProvider` — Chat Completions API, streaming.
- `AnthropicProvider` — Messages API (Claude models), streaming.
- `GoogleProvider` — Generative Language API (Gemini models), streaming.
- `AzureProvider` — Azure OpenAI Chat Completions (deployment-based URLs).
- `BedrockProvider` — AWS Bedrock (Anthropic + Amazon Titan), async via thread executor.
- `OllamaProvider` — OpenAI-compatible local inference, cost = 0.
- `GroqProvider` — OpenAI-compatible cloud inference.
- `LMStudioProvider` — OpenAI-compatible local inference, cost = 0.

#### Semantic cache

- `SemanticCache` — vector-similarity-based response cache (threshold, TTL, tags, namespaces).
- Backends: `InMemoryBackend`, `RedisBackend`, `ChromaBackend`, `QdrantBackend`, `PineconeBackend`.
- Encoders: `OpenAIEncoder` (text-embedding-3-small), `SentenceTransformerEncoder` (local/offline).
- `CacheEncryption` — AES-256-GCM encryption for cached responses.
- Tag-based invalidation (`invalidate_by_tag`).

#### Model router

- `ModelRouter` with four strategies: `complexity_based`, `cost_optimized`, `latency_optimized`, `round_robin`.
- `CostCalculator` — per-request cost estimation from provider price tables.
- Custom strategy support via callable `(RequestContext) -> RouteResult`.
- Router integration into `LLMClient` (auto-routes when no explicit model is passed).

#### Resilience

- `ResilienceEngine` — configurable retry (exponential backoff), timeout, circuit breaker.
- `CircuitBreaker` — failure threshold, half-open recovery.
- `FallbackChain` — ordered provider fallback with aggregated error reporting.
- `RateLimiter` middleware — token-bucket per-provider rate limiting.

#### Middleware pipeline

- `BaseMiddleware` — `before_request` / `after_response` / `on_error` hooks.
- `PromptInjectionGuard` — blocks common prompt injection patterns.
- `MetricsCollector` — 11 metrics with rolling window, percentiles, and callbacks.

#### Observability

- `MetricsCollector` — tracks 11 metrics (tokens, cost, cache hits, latency P50/P95/P99, error types, calls by model/provider).
- `PrometheusExporter` — delta-sync counters + gauges, optional HTTP scrape server.
- `OTLPExporter` — OpenTelemetry SDK push to any OTLP-compatible backend.
- `grafana_dashboard.json` — ready-to-import Grafana dashboard.

#### Configuration

- `LLMConfig.from_env()` — reads 15+ environment variables across all providers.
- `LLMConfig.from_file()` — YAML (pyyaml) and TOML (tomllib / tomli) support.
- `VaultConfig` — HashiCorp Vault KV v2 with `hvac`, merges secrets into base config.

#### CI/CD and tooling

- GitHub Actions: lint + mypy + test matrix (Python 3.10/3.11/3.12 × Linux/macOS/Windows).
- Security scan: `bandit` + `pip-audit` on push + weekly cron.
- Auto-release to PyPI on `v*` tag via OIDC trusted publishing.
- `docker-compose.yml` for Redis + ChromaDB integration tests.

#### Documentation

- `README.md` — quick start, badges, provider table, code examples.
- `docs/getting_started.md` — Hello World in under 10 minutes.
- `docs/cookbook.md` — 20 recipes.
- `docs/configuration_reference.md` — all fields, env vars, YAML/TOML/Vault.
- `docs/migration_guide.md` — from OpenAI SDK, Anthropic SDK, LiteLLM, LangChain.
- `docs/adr/` — 4 Architecture Decision Records.
- `CONTRIBUTING.md` — setup, testing, PR checklist.
- Sphinx API docs for all public classes and methods.

### Tests

- 342 unit tests, 91.28% coverage.
- Integration tests: Redis backend, ChromaDB backend.
- Performance benchmark: Redis cache lookup P95 ≤ 20ms at 100k entries.

---

## [0.1.0] — Unreleased

_Initial project scaffold — no public API stable._

[1.0.0]: https://github.com/your-org/llm-std-lib/releases/tag/v1.0.0
