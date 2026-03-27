# ADR-004: Use httpx directly instead of provider SDKs

**Status:** Accepted
**Date:** 2025-02-01

## Context

Each provider ships its own Python SDK (`openai`, `anthropic`, `google-generativeai`, etc.).
Depending on all of them would:
1. Inflate the install footprint by hundreds of MB.
2. Create conflicting dependency trees.
3. Make the library vulnerable to breaking changes in 8+ separate packages.

## Decision

All provider adapters use `httpx.AsyncClient` directly and implement the REST APIs
themselves. The only exception is AWS Bedrock, which requires `boto3` for AWS credential
chain signing (SigV4) — boto3 is listed as an optional extra.

## Consequences

- **Positive:** Core install is ~5 packages; fast to install.
- **Positive:** Full control over request/response handling and error mapping.
- **Negative:** Must track each provider's API changes manually.
- **Negative:** AWS Bedrock requires boto3 (synchronous); wrapped in `run_in_executor`.
