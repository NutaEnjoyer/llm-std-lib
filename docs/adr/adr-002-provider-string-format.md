# ADR-002: provider/model string convention

**Status:** Accepted
**Date:** 2025-01-15

## Context

Users need a concise way to specify both provider and model in a single argument, consistent
with established community conventions (LiteLLM, OpenRouter).

## Decision

Models are identified by `"provider/model"` strings, e.g. `"openai/gpt-4o-mini"`.
The `LLMClient` splits on the first `/` to resolve provider and model.
If no `/` is present, the default provider is used.

## Consequences

- **Positive:** Single string replaces two separate parameters; familiar to LiteLLM users.
- **Positive:** Router strategies can return `RouteResult(provider, model)` directly.
- **Negative:** Provider names must not contain `/`; documented as a constraint.
