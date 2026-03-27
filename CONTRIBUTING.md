# Contributing to llm-std-lib

Thank you for your interest in contributing! This document covers everything you need
to get started.

## Development setup

```bash
git clone https://github.com/your-org/llm-std-lib
cd llm-std-lib
pip install -e ".[all]"
pip install ruff mypy pytest pytest-asyncio pytest-cov
```

## Running tests

```bash
# Unit tests (no external services required)
pytest tests/unit/ -q

# With coverage
pytest tests/unit/ --cov=llm_std_lib --cov-report=term-missing

# Integration tests (requires Docker)
docker compose up -d
pytest tests/integration/ -q
docker compose down
```

## Code quality

Before opening a PR, make sure these pass:

```bash
# Linting and formatting
python -m ruff check llm_std_lib
python -m ruff format llm_std_lib   # or black

# Type checking
python -m mypy llm_std_lib --ignore-missing-imports --no-namespace-packages
```

The CI pipeline enforces all three. A PR with ruff or mypy failures will not be merged.

## Adding a new provider

1. Create `llm_std_lib/providers/<name>.py` by subclassing `BaseProvider`.
2. Implement `complete()` and `stream()`.
3. Register the provider name in `LLMClient._build_provider()` (`client.py`).
4. Add pricing to `PROVIDER_PRICES` in `config.py` (if applicable).
5. Add `from_env()` support in `LLMConfig.from_env()`.
6. Write tests in `tests/unit/test_providers.py` — aim for ≥90% coverage.

## Adding a new cache backend

1. Create `llm_std_lib/cache/backends/<name>.py` by subclassing `BaseCacheBackend`.
2. Implement all six abstract methods: `search`, `store`, `delete`, `clear`,
   `invalidate_by_tag`, `size`.
3. Add the optional dependency to `pyproject.toml`.
4. Write integration tests in `tests/integration/`.

## Commit style

We follow conventional commits:

```
feat: add Cohere provider adapter
fix: handle empty response from Bedrock Titan
docs: add cookbook recipe for batch processing
test: add integration tests for Qdrant backend
refactor: extract _parse_oai_response helper
```

## Pull request checklist

- [ ] Tests pass (`pytest tests/unit/`)
- [ ] Coverage ≥90% for changed modules
- [ ] `ruff check` passes
- [ ] `mypy` passes
- [ ] Docstrings added for all new public classes and methods
- [ ] `CHANGELOG.md` updated
- [ ] PR description explains *why*, not just *what*

## Reporting bugs

Open an issue with:
- Python version and OS
- llm-std-lib version (`pip show llm-std-lib`)
- Minimal reproducible example
- Full traceback

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
